use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt::{Debug, Display, Formatter};
use std::io;
use std::string::FromUtf8Error;
use once_cell::sync::Lazy;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum AllowedSpecial<'a> {
    All,
    Allowed(HashSet<&'a str>),
}

#[derive(Debug, Clone)]
pub enum DisallowedSpecial<'a> {
    All,
    Disallowed(HashSet<&'a str>),
}

pub enum DecodeMode {
    Strict,
    Replace, // replace invalid character
}

#[derive(Debug, Error)]
pub enum EncodeError {
    #[error("regex error: {0}")]
    RegexError(#[from] fancy_regex::Error),
    #[error("token `{0}` not found")]
    TokenNotFoundError(usize),
    #[error("could not encode `{0:?}` to token")]
    TokenEncodeError(Vec<u8>),
    #[error("Encountered text corresponding to disallowed special token '{0}'.\n
If you want this text to be encoded as a special token, pass it to `allowed_special`.\n
If you want this text to be encoded as normal text, disable the check for this token \
by passing `disallowed_special=(enc.special_tokens_set - {{'{0}'}})`.\n
To disable this check for all special tokens, pass `disallowed_special=()`.\n")]
    SpecialTokenError(String),
    #[error("convert bytes to string error: {0}")]
    ConvertStringError(#[from] FromUtf8Error),
    #[error("Could not automatically map {0} to a tokeniser.
Please use `tiktoken.get_encoding` to explicitly get the tokeniser you expect.")]
    ModelNameError(String),
    #[error("Unknown encoding {0}")]
    EncodingNameError(String),
    #[error("Stdio error: {0}")]
    IOError(#[from] io::Error),
    #[error("Network error: {0}")]
    HTTPError(#[from] reqwest::Error),
}


// TODO: these will likely be replaced by an API endpoint
pub static MODEL_PREFIX_TO_ENCODING: Lazy<HashMap<&str, &str>> = Lazy::new(|| {
    HashMap::from([
        // chat
        ("gpt-4-", "cl100k_base"), // e.g., gpt-4-0314, etc., plus gpt-4-32k
        ("gpt-3.5-turbo-", "cl100k_base"), // e.g, gpt-3.5-turbo-0301, -0401, etc.
    ])
});


pub static MODEL_TO_ENCODING: Lazy<HashMap<&str, &str>> = Lazy::new(|| {
    HashMap::from([
        // chat
        ("gpt-4", "cl100k_base"),
        ("gpt-3.5-turbo", "cl100k_base"),
        // text
        ("text-davinci-003", "p50k_base"),
        ("text-davinci-002", "p50k_base"),
        ("text-davinci-001", "r50k_base"),
        ("text-curie-001", "r50k_base"),
        ("text-babbage-001", "r50k_base"),
        ("text-ada-001", "r50k_base"),
        ("davinci", "r50k_base"),
        ("curie", "r50k_base"),
        ("babbage", "r50k_base"),
        ("ada", "r50k_base"),
        // code
        ("code-davinci-002", "p50k_base"),
        ("code-davinci-001", "p50k_base"),
        ("code-cushman-002", "p50k_base"),
        ("code-cushman-001", "p50k_base"),
        ("davinci-codex", "p50k_base"),
        ("cushman-codex", "p50k_base"),
        // edit
        ("text-davinci-edit-001", "p50k_edit"),
        ("code-davinci-edit-001", "p50k_edit"),
        // embeddings
        ("text-embedding-ada-002", "cl100k_base"),
        // old embeddings
        ("text-similarity-davinci-001", "r50k_base"),
        ("text-similarity-curie-001", "r50k_base"),
        ("text-similarity-babbage-001", "r50k_base"),
        ("text-similarity-ada-001", "r50k_base"),
        ("text-search-davinci-doc-001", "r50k_base"),
        ("text-search-curie-doc-001", "r50k_base"),
        ("text-search-babbage-doc-001", "r50k_base"),
        ("text-search-ada-doc-001", "r50k_base"),
        ("code-search-babbage-code-001", "r50k_base"),
        ("code-search-ada-code-001", "r50k_base"),
        // open source
        ("gpt2", "gpt2"),
    ])
});
