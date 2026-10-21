//! Atomic begin/image/end spans with causal text positions and spatial feature order.

use crate::{
    Result,
    cache::ConditioningEvent,
    frontend::{
        Frontend, RenderedPrompt,
        images::{ImageDescription, ImageProcessor},
    },
    model::{PreparedInput, PromptData},
};

pub(crate) fn prepare_input(
    frontend: &Frontend,
    processor: Option<&ImageProcessor>,
    rendered: &RenderedPrompt,
    images: &[ImageDescription],
) -> Result<PreparedInput> {
    let atomic = |text| -> Result<u32> {
        let ids = frontend.encode(text, false)?;
        match ids.as_slice() {
            [id] => Ok(*id),
            _ => Err("GLM media markers must be atomic tokenizer tokens".into()),
        }
    };
    let start = atomic("<|begin_of_image|>")?;
    let feature = atomic("<|image|>")?;
    let end = atomic("<|end_of_image|>")?;
    let video = [
        atomic("<|begin_of_video|>")?,
        atomic("<|video|>")?,
        atomic("<|end_of_video|>")?,
    ];
    if [start, feature, end]
        .iter()
        .collect::<std::collections::HashSet<_>>()
        .len()
        != 3
    {
        return Err("GLM image markers must have distinct tokenizer IDs".into());
    }
    if rendered.token_ids.iter().any(|id| video.contains(id)) {
        return Err("video inputs are not supported by the GLM image frontend".into());
    }
    let source_tokens = &rendered.token_ids;
    for (index, id) in source_tokens.iter().enumerate() {
        let valid = if *id == start {
            source_tokens.get(index + 1) == Some(&feature)
                && source_tokens.get(index + 2) == Some(&end)
        } else if *id == feature {
            index > 0
                && source_tokens[index - 1] == start
                && source_tokens.get(index + 1) == Some(&end)
        } else if *id == end {
            index >= 2 && source_tokens[index - 1] == feature && source_tokens[index - 2] == start
        } else {
            true
        };
        if !valid {
            return Err(
                "GLM image input requires one begin/image/end placeholder per image".into(),
            );
        }
    }
    if source_tokens.iter().filter(|id| **id == feature).count() != images.len() {
        return Err("GLM image placeholders do not match input images".into());
    }
    if !images.is_empty() {
        processor
            .ok_or("image processing is unavailable")?
            .describe_inputs(images)?;
    }
    let mut tokens = Vec::new();
    let mut data = PromptData::default();
    let mut conditioning = Vec::new();
    let mut token_expansions = Vec::new();
    let (mut image_index, mut feature_offset) = (0, 0usize);
    for (index, id) in source_tokens.iter().enumerate() {
        if *id != feature {
            tokens.push(*id);
            data.vision_token_indices.push(-1);
            continue;
        }
        let image = &images[image_index];
        let count = image.geometry.tokens;
        let next = feature_offset
            .checked_add(count)
            .filter(|n| *n <= i32::MAX as usize)
            .ok_or("GLM feature indices exceed int32")?;
        if count == 0 {
            return Err("GLM image has no feature tokens".into());
        }
        let position = tokens.len();
        tokens.extend(std::iter::repeat_n(feature, count));
        data.vision_token_indices
            .extend((feature_offset..next).map(|n| n as i32));
        data.spans.push(position - 1..position + count + 1);
        conditioning.push(ConditioningEvent {
            position: position as u64,
            identity: image.identity,
        });
        token_expansions.push((index, count - 1));
        image_index += 1;
        feature_offset = next;
    }
    Ok(PreparedInput {
        tokens,
        token_expansions,
        data,
        conditioning,
        conditioned_inputs: vec![],
    })
}
