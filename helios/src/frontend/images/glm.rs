//! Upward-aligned, aspect-preserving GLM image canvases and temporal patch packing.

use super::*;

pub(super) const PATCH: usize = 14;
pub(super) const MERGE: usize = 2;
pub(super) const PIXEL_WIDTH: usize = 3 * 2 * PATCH * PATCH;
const MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

#[derive(Clone)]
pub(super) struct GlmProcessor {
    limits: ImageLimits,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
}

impl GlmProcessor {
    pub(super) fn load(settings: GlmImageSettings, limits: ImageLimits) -> Result<Self> {
        if settings.min_image_tokens == 0
            || settings.max_image_tokens < settings.min_image_tokens
            || settings.patch_expand_factor == 0
            || limits.patches_per_feature != MERGE * MERGE
        {
            return Err("invalid prepared GLM image settings or patch-to-feature ratio".into());
        }
        let factor = (PATCH * MERGE)
            .checked_mul(settings.patch_expand_factor)
            .ok_or("GLM image alignment overflows")?;
        let pixels_per_token = 2 * (PATCH * MERGE).pow(2);
        let min_pixels = settings
            .min_image_tokens
            .checked_mul(pixels_per_token)
            .ok_or("GLM minimum image budget overflows")?;
        let max_pixels = settings
            .max_image_tokens
            .checked_mul(pixels_per_token)
            .ok_or("GLM maximum image budget overflows")?;
        if (factor as u128).pow(2) > max_pixels as u128 / 2 {
            return Err("GLM image budget cannot hold one aligned patch group".into());
        }
        Ok(Self {
            limits,
            factor,
            min_pixels,
            max_pixels,
        })
    }

    fn aligned(&self, value: usize) -> Result<usize> {
        value
            .checked_add(self.factor - 1)
            .map(|n| n / self.factor * self.factor)
            .ok_or_else(|| "GLM image alignment overflows".into())
    }

    fn fit(&self, height: usize, width: usize) -> Result<(usize, usize)> {
        let (mut low, mut high) = (1, height);
        let (mut best_h, mut best_w) = (self.factor, self.factor);
        while low <= high {
            let content_h = low + (high - low) / 2;
            let content_w =
                ((width as f64 * content_h as f64 / height as f64).floor() as usize).max(1);
            let (h, w) = (self.aligned(content_h)?, self.aligned(content_w)?);
            if 2 * h as u128 * w as u128 <= self.max_pixels as u128 {
                (best_h, best_w) = (h, w);
                low = content_h + 1;
            } else {
                high = content_h - 1;
            }
        }
        Ok((best_h, best_w))
    }

    pub(super) fn geometry(&self, width: usize, height: usize) -> Result<ImageGeometry> {
        check_input_extent(self.limits, width, height)?;
        let (mut h, mut w) = (self.aligned(height)?, self.aligned(width)?);
        let pixels = 2 * h as u128 * w as u128;
        if pixels > self.max_pixels as u128 {
            (h, w) = self.fit(height, width)?;
        } else if pixels < self.min_pixels as u128 {
            let scale = (self.min_pixels as f64 / (2. * height as f64 * width as f64)).sqrt();
            h = self.aligned((height as f64 * scale).ceil() as usize)?;
            w = self.aligned((width as f64 * scale).ceil() as usize)?;
            if 2 * h as u128 * w as u128 > self.max_pixels as u128 {
                (h, w) = self.fit(height, width)?;
            }
        }
        let patches = (h / PATCH)
            .checked_mul(w / PATCH)
            .ok_or("GLM patch count overflows")?;
        check_image_patches(self.limits, patches)?;
        Ok(ImageGeometry {
            height: h,
            width: w,
            grid_thw: [1, (h / PATCH) as i64, (w / PATCH) as i64],
            patches,
            tokens: patches / (MERGE * MERGE),
        })
    }

    pub(super) fn resized_rgb(&self, image: &RgbImage) -> Result<(ImageGeometry, Vec<u8>)> {
        let geometry = self.geometry(image.width, image.height)?;
        let mut scale = (geometry.height as f64 / image.height as f64)
            .min(geometry.width as f64 / image.width as f64);
        if 2 * image.height as u128 * image.width as u128 >= self.min_pixels as u128 {
            scale = scale.min(1.);
        }
        let h = ((image.height as f64 * scale).floor() as usize).clamp(1, geometry.height);
        let w = ((image.width as f64 * scale).floor() as usize).clamp(1, geometry.width);
        let content = resize::resize(image, w, h);
        let nbytes = geometry
            .height
            .checked_mul(geometry.width)
            .and_then(|n| n.checked_mul(3))
            .ok_or("GLM canvas extent overflows")?;
        let mut canvas = vec![0; nbytes];
        for y in 0..h {
            canvas[y * geometry.width * 3..(y * geometry.width + w) * 3]
                .copy_from_slice(&content[y * w * 3..(y + 1) * w * 3]);
        }
        Ok((geometry, canvas))
    }

    pub(super) fn process(&self, images: &[RgbImage]) -> Result<ProcessedImages> {
        self.limits.check_batch(images.len(), 0)?;
        let geometry = images
            .iter()
            .map(|image| self.geometry(image.width, image.height))
            .collect::<Result<Vec<_>>>()?;
        let total = geometry
            .iter()
            .try_fold(0usize, |n, image| n.checked_add(image.patches))
            .ok_or("GLM patch count overflows")?;
        self.limits.check_batch(images.len(), total)?;
        let mut result = ProcessedImages {
            geometry,
            cu_seqlens: vec![0],
            ..Default::default()
        };
        result.reserve_outputs(total, PIXEL_WIDTH, false, images.len(), images.len() + 1)?;
        let normalized: [[f32; 256]; 3] = std::array::from_fn(|channel| {
            // Match BaseImageProcessorFast's fused rescale/normalize: fuse in
            // float32 before scaling by 255, then normalize uint8->float32 pixels.
            let mean = MEAN[channel] * 255.;
            let std = STD[channel] * 255.;
            std::array::from_fn(|value| (value as f32 - mean) / std)
        });
        for (image, geometry) in images.iter().zip(&result.geometry) {
            let (_, pixels) = self.resized_rgb(image)?;
            for by in 0..geometry.height / PATCH / MERGE {
                for bx in 0..geometry.width / PATCH / MERGE {
                    for dy in 0..MERGE {
                        for dx in 0..MERGE {
                            let (row, col) = (by * MERGE + dy, bx * MERGE + dx);
                            result.position_ids.push([row as i64, col as i64]);
                            for channel in 0..3 {
                                for _temporal in 0..2 {
                                    for y in 0..PATCH {
                                        for x in 0..PATCH {
                                            let value = pixels[((row * PATCH + y)
                                                * geometry.width
                                                + col * PATCH
                                                + x)
                                                * 3
                                                + channel];
                                            result
                                                .pixel_values
                                                .push(normalized[channel][value as usize]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            result
                .cu_seqlens
                .push(result.cu_seqlens.last().unwrap() + geometry.patches as i32);
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn processor() -> ImageProcessor {
        ImageProcessor::load(
            ImageSettings::Glm(GlmImageSettings {
                min_image_tokens: 16,
                max_image_tokens: 2048,
                patch_expand_factor: 1,
            }),
            ImageLimits {
                max_patches_per_image: 8192,
                patches_per_feature: 4,
                ..Default::default()
            },
        )
        .unwrap()
    }

    #[test]
    fn canvases_preserve_aspect_and_pad_right_and_bottom() {
        let p = processor();
        let image = RgbImage::new(29, 30, vec![255; 29 * 30 * 3]).unwrap();
        let (g, pixels) = p.resized_rgb(&image).unwrap();
        assert_eq!((g.height, g.width, g.grid_thw), (140, 112, [1, 10, 8]));
        assert_eq!(g.tokens, 20);
        assert_eq!(&pixels[..3], &[255; 3]);
        assert_eq!(&pixels[(130 * 112) * 3..(130 * 112 + 1) * 3], &[0; 3]);
        assert!(p.geometry(1, 0).is_err());
        let large = p.geometry(4000, 3000).unwrap();
        assert!(large.tokens <= 2048);
        assert_eq!((large.height % 28, large.width % 28), (0, 0));
    }

    #[test]
    fn packed_patches_duplicate_time_and_keep_merge_order() {
        let p = processor();
        let mut rgb = vec![0; 112 * 112 * 3];
        for y in 0..112 {
            for x in 0..112 {
                rgb[(y * 112 + x) * 3] = (y / 14 * 8 + x / 14) as u8;
            }
        }
        let images = p.process(&[RgbImage::new(112, 112, rgb).unwrap()]).unwrap();
        assert_eq!(images.geometry[0].patches, 64);
        assert_eq!(
            images.position_ids[..8],
            [
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
                [0, 2],
                [0, 3],
                [1, 2],
                [1, 3]
            ]
        );
        assert_eq!(images.pixel_values.len(), 64 * PIXEL_WIDTH);
        assert_eq!(&images.pixel_values[..196], &images.pixel_values[196..392]);
        assert_eq!(images.cu_seqlens, [0, 64]);
        let restored = ProcessedImages::decode(&images.encode().unwrap()).unwrap();
        assert_eq!(images.pixel_values, restored.pixel_values);
        assert_eq!(images.position_ids, restored.position_ids);
    }

    #[test]
    fn invalid_settings_and_budgets_fail_before_allocating_pixels() {
        for settings in [
            GlmImageSettings {
                min_image_tokens: 0,
                max_image_tokens: 16,
                patch_expand_factor: 1,
            },
            GlmImageSettings {
                min_image_tokens: 16,
                max_image_tokens: 15,
                patch_expand_factor: 1,
            },
            GlmImageSettings {
                min_image_tokens: 16,
                max_image_tokens: 16,
                patch_expand_factor: 0,
            },
            GlmImageSettings {
                min_image_tokens: 1,
                max_image_tokens: 1,
                patch_expand_factor: 2,
            },
        ] {
            assert!(
                ImageProcessor::load(
                    ImageSettings::Glm(settings),
                    ImageLimits {
                        patches_per_feature: 4,
                        ..Default::default()
                    }
                )
                .is_err()
            );
        }
        assert!(
            ImageProcessor::load(
                ImageSettings::Glm(GlmImageSettings {
                    min_image_tokens: 16,
                    max_image_tokens: 2048,
                    patch_expand_factor: 1,
                }),
                ImageLimits::default()
            )
            .is_err()
        );
        assert!(processor().process(&[]).is_err());
    }
}
