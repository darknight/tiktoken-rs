use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, Mutex};
use once_cell::sync::Lazy;
use crate::core::EncodingParam;
use crate::load::{data_gym_to_mergeable_bpe_ranks, load_tiktoken_bpe};

const ENDOFTEXT: &str = "<|endoftext|>";
const FIM_PREFIX: &str = "<|fim_prefix|>";
const FIM_MIDDLE: &str = "<|fim_middle|>";
const FIM_SUFFIX: &str = "<|fim_suffix|>";
const ENDOFPROMPT: &str = "<|endofprompt|>";

static ENCODING_TO_CONSTRUCTOR: Lazy<HashMap<&'static str, Box<dyn Fn() -> EncodingParam + Send + Sync>>> = Lazy::new(|| {
    let mut m = HashMap::new();
    m.insert("gpt2", Box::new(gpt2) as Box<dyn Fn() -> EncodingParam + Send + Sync>);
    m.insert("r50k_base", Box::new(r50k_base) as Box<dyn Fn() -> EncodingParam + Send + Sync>);
    m.insert("p50k_base", Box::new(p50k_base) as Box<dyn Fn() -> EncodingParam + Send + Sync>);
    m.insert("p50k_edit", Box::new(p50k_edit) as Box<dyn Fn() -> EncodingParam + Send + Sync>);
    m.insert("cl100k_base", Box::new(cl100k_base) as Box<dyn Fn() -> EncodingParam + Send + Sync>);
    m
});


pub fn find_encoding_constructor(encoding_name: &str) -> Option<&Box<dyn Fn() -> EncodingParam + Send + Sync>> {
    ENCODING_TO_CONSTRUCTOR.get(encoding_name)
}

/// List available encodings by name
pub fn list_encoding_names<'a>() -> Vec<&'a str> {
    ENCODING_TO_CONSTRUCTOR.keys().copied().collect()
}

fn gpt2() -> EncodingParam {
    let mergeable_ranks = data_gym_to_mergeable_bpe_ranks(
        "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe",
        "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json"
    );
    let special_tokens = vec![
        (ENDOFTEXT.to_string(), 50256usize)
    ];

    EncodingParam::new(
        "gpt2".to_string(),
        r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+".to_string(),
        mergeable_ranks,
        special_tokens.into_iter().collect(),
        Some(50257)
    )
}

fn r50k_base() -> EncodingParam {
    let mergeable_ranks = load_tiktoken_bpe(
        "https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken"
    );
    let special_tokens = vec![
        (ENDOFTEXT.to_string(), 50256usize)
    ];

    EncodingParam::new(
        "r50k_base".to_string(),
        r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+".to_string(),
        mergeable_ranks,
        special_tokens.into_iter().collect(),
        Some(50257)
    )
}

fn p50k_base() -> EncodingParam {
    let mergeable_ranks = load_tiktoken_bpe(
        "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken"
    );
    let special_tokens = vec![
        (ENDOFTEXT.to_string(), 50256usize)
    ];

    EncodingParam::new(
        "p50k_base".to_string(),
        r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+".to_string(),
        mergeable_ranks,
        special_tokens.into_iter().collect(),
        Some(50281)
    )
}

fn p50k_edit() -> EncodingParam {
    let mergeable_ranks = load_tiktoken_bpe(
        "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken"
    );
    let special_tokens = vec![
        (ENDOFTEXT.to_string(), 50256usize),
        (FIM_PREFIX.to_string(), 50281usize),
        (FIM_MIDDLE.to_string(), 50282usize),
        (FIM_SUFFIX.to_string(), 50283usize),
    ];

    EncodingParam::new(
        "p50k_edit".to_string(),
        r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+".to_string(),
        mergeable_ranks,
        special_tokens.into_iter().collect(),
        None
    )
}

fn cl100k_base() -> EncodingParam {
    let mergeable_ranks = load_tiktoken_bpe(
        "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
    );
    let special_tokens = vec![
        (ENDOFTEXT.to_string(), 100257usize),
        (FIM_PREFIX.to_string(), 100258usize),
        (FIM_MIDDLE.to_string(), 100259usize),
        (FIM_SUFFIX.to_string(), 100260usize),
        (ENDOFPROMPT.to_string(), 100276usize),
    ];

    EncodingParam::new(
        "cl100k_base".to_string(),
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+".to_string(),
        mergeable_ranks,
        special_tokens.into_iter().collect(),
        None
    )
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_encoding_names() {
        let mut res = list_encoding_names();
        res.sort();

        assert_eq!(res, vec!["cl100k_base", "gpt2", "p50k_base", "p50k_edit", "r50k_base"]);
    }
}