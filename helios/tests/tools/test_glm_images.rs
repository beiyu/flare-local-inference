//! Exact CPU preparation parity against the deployment's real GLM processor.
use flare_helios::frontend::images::{
    GlmImageSettings, ImageLimits, ImageProcessor, ImageSettings, RgbImage,
};
use serde_json::Value;
use sha2::{Digest, Sha256};

fn digest(bytes: impl IntoIterator<Item = u8>) -> String {
    let bytes = bytes.into_iter().collect::<Vec<_>>();
    format!("{:x}", Sha256::digest(bytes))
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: test_glm_images REFERENCE.json");
    let reference: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let cases = reference["cases"]
        .as_array()
        .expect("missing reference cases");
    assert!(
        !cases.is_empty(),
        "reference must contain real processor cases"
    );
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let settings: GlmImageSettings = serde_json::from_value(case["settings"].clone()).unwrap();
        let processor = ImageProcessor::load(
            ImageSettings::Glm(settings),
            ImageLimits {
                max_patches_per_image: 8192,
                patches_per_feature: 4,
                ..Default::default()
            },
        )
        .unwrap();
        let images = case["inputs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|input| {
                let width = input["width"].as_u64().unwrap() as usize;
                let height = input["height"].as_u64().unwrap() as usize;
                let mut state = input["seed"].as_u64().unwrap() as u32;
                let pixels = (0..width * height * 3)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 17;
                        state ^= state << 5;
                        state as u8
                    })
                    .collect();
                RgbImage::new(width, height, pixels).unwrap()
            })
            .collect::<Vec<_>>();
        for (index, image) in images.iter().enumerate() {
            let (geometry, pixels) = processor.resized_rgb(image).unwrap();
            assert_eq!(
                serde_json::json!(geometry.grid_thw),
                case["grid_thw"][index],
                "{name}: grid"
            );
            assert_eq!(
                digest(pixels),
                case["resized_rgb_sha256"][index].as_str().unwrap(),
                "{name}: canvas {index}"
            );
        }
        let processed = processor.process(&images).unwrap();
        let elements: u64 = case["pixel_values_shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .product();
        assert_eq!(
            processed.pixel_values.len() as u64,
            elements,
            "{name}: pixel shape"
        );
        assert_eq!(
            digest(
                processed
                    .pixel_values
                    .iter()
                    .copied()
                    .flat_map(f32::to_le_bytes)
            ),
            case["pixel_values_sha256"].as_str().unwrap(),
            "{name}: normalized patches"
        );
        assert_eq!(
            digest(
                processed
                    .position_ids
                    .iter()
                    .flatten()
                    .copied()
                    .flat_map(i64::to_le_bytes)
            ),
            case["position_ids_sha256"].as_str().unwrap(),
            "{name}: vision positions"
        );
        assert_eq!(
            serde_json::json!(processed.cu_seqlens),
            case["cu_seqlens"],
            "{name}: sequence boundaries"
        );
        println!("{name}: exact CPU frontend parity");
    }
    println!(
        "{} GLM CPU image preparation cases passed; vision encoder serving is not qualified",
        cases.len()
    );
}
