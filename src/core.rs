use std::cmp::max;
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display, Error, format, Formatter, Pointer};
use std::hash::Hash;
use std::io::Read;
use std::string::FromUtf8Error;
use fancy_regex::Regex;
use crate::CoreBPE;
use crate::model::*;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use crate::openai_public::find_encoding_constructor;

pub type Result<T> = std::result::Result<T, EncodeError>;


/// Return Encoding object
/// TODO: cache created Encoding object
pub fn get_encoding(encoding_name: &str) -> Result<Encoding> {
    match find_encoding_constructor(encoding_name) {
        Some(func) => {
            Encoding::new(func())
        },
        None => Err(EncodeError::EncodingNameError(encoding_name.to_string())),
    }
}

/// Returns the encoding used by a model.
pub fn encoding_for_model(model_name: &str) -> Result<Encoding> {
    let encoding_opt = MODEL_TO_ENCODING.get(model_name)
        .map(|&encoding| get_encoding(encoding));
    if encoding_opt.is_some() {
        return encoding_opt.unwrap();
    }

    // Check if the model matches a known prefix
    // Prefix matching avoids needing library updates for every model version release
    // Note that this can match on non-existent models (e.g., gpt-3.5-turbo-FAKE)
    for (&model_prefix, &model_encoding_name) in MODEL_PREFIX_TO_ENCODING.iter() {
        if model_name.starts_with(model_prefix) {
            return get_encoding(model_encoding_name);
        }
    }

    Err(EncodeError::ModelNameError(model_name.to_string()))
}

pub struct EncodingParam {
    name: String,
    pat_str: String,
    mergeable_ranks: HashMap<Vec<u8>, usize>,
    special_tokens: HashMap<String, usize>,
    explicit_n_vocab: Option<usize>,
}

impl EncodingParam {
    pub fn new(name: String,
               pat_str: String,
               mergeable_ranks: HashMap<Vec<u8>, usize>,
               special_tokens: HashMap<String, usize>,
               explicit_n_vocab: Option<usize>) -> Self {
        EncodingParam {
            name,
            pat_str,
            mergeable_ranks,
            special_tokens,
            explicit_n_vocab,
        }
    }
}

pub struct Encoding {
    name: String,
    pat_str: String,
    special_tokens: HashMap<String, usize>,

    max_token_value: usize,
    core_bpe: CoreBPE,
}

impl Debug for Encoding {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "<Encoding '{}'>", self.name)
    }
}

/// Display
impl Display for Encoding {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "<Encoding '{}'>", self.name)
    }
}

/// Private methods
impl Encoding {
    ///
    /// Creates an Encoding object.
    ///
    /// See openai_public.py for examples of how to construct an Encoding object.
    ///
    /// Args:
    /// name: The name of the encoding. It should be clear from the name of the encoding
    ///       what behaviour to expect, in particular, encodings with different special tokens
    ///       should have different names.
    /// pat_str: A regex pattern string that is used to split the input text.
    /// mergeable_ranks: A dictionary mapping mergeable token bytes to their ranks. The ranks
    ///                  must correspond to merge priority.
    /// special_tokens: A dictionary mapping special token strings to their token values.
    /// explicit_n_vocab: The number of tokens in the vocabulary. If provided, it is checked
    ///                   that the number of mergeable tokens and special tokens is equal to this number.
    ///
    fn new(param: EncodingParam) -> Result<Self> {

        let max_token_value = max(
            param.mergeable_ranks.values().max().copied().unwrap_or_default(),
            param.special_tokens.values().max().copied().unwrap_or_default()
        );
        if let Some(n_vocab) = param.explicit_n_vocab {
            assert_eq!(param.mergeable_ranks.len() + param.special_tokens.len(), n_vocab);
            assert_eq!(max_token_value, n_vocab - 1);
        }

        let core_bpe = CoreBPE::new(
            convert_to_fx_hashmap(&param.mergeable_ranks),
            convert_to_fx_hashmap(&param.special_tokens),
            param.pat_str.as_str())?;

        Ok(Encoding {
            name: param.name,
            pat_str: param.pat_str,
            special_tokens: param.special_tokens,
            max_token_value,
            core_bpe
        })
    }
}

/// Public interfaces for encoding
impl Encoding {

    ///Encodes a string into tokens, ignoring special tokens.
    ///
    /// This is equivalent to `encode(text, disallowed_special=())` (but slightly faster).
    pub fn encode_ordinary(&self, text: &str) -> Vec<usize> {
        self.core_bpe._encode_ordinary_native(text)
    }

    ///Encodes a list of strings into tokens, in parallel, ignoring special tokens.
    ///
    /// This is equivalent to `encode_batch(text, disallowed_special=())` (but slightly faster).
    pub fn encode_ordinary_batch(&self, texts: Vec<&str>) -> Vec<Vec<usize>> {
        texts.par_iter()
            .map(|&txt| self.encode_ordinary(txt))
            .collect()
    }

    /// Encodes a string into tokens.
    /// Special tokens are artificial tokens used to unlock capabilities from a model,
    /// such as fill-in-the-middle. So we want to be careful about accidentally encoding special
    /// tokens, since they can be used to trick a model into doing something we don't want it to do.
    /// Hence, by default, encode will raise an error if it encounters text that corresponds
    /// to a special token. This can be controlled on a per-token level using the `allowed_special`
    /// and `disallowed_special` parameters. In particular:
    /// - Setting `disallowed_special` to () will prevent this function from raising errors and
    /// cause all text corresponding to special tokens to be encoded as natural text.
    /// - Setting `allowed_special` to "All" will cause this function to treat all text
    /// corresponding to special tokens to be encoded as special tokens.
    pub fn encode(&self,
                  text: &str,
                  allowed_special: AllowedSpecial,
                  disallowed_special: DisallowedSpecial) -> Result<Vec<usize>> {

        let allowed_special_set = match allowed_special {
            AllowedSpecial::All => self.special_tokens_set(),
            AllowedSpecial::Allowed(allowed) => allowed,
        };
        let disallowed_special_set = match disallowed_special {
            DisallowedSpecial::All =>
                self.special_tokens_set()
                    .difference(&allowed_special_set)
                    .copied()
                    .collect(),
            DisallowedSpecial::Disallowed(disallowed) => disallowed,
        };

        if !disallowed_special_set.is_empty() {
            let re = special_token_regex(disallowed_special_set)?;
            if let Ok(Some(cap)) = re.captures(text) {
                return Err(EncodeError::SpecialTokenError(
                    String::from(cap.get(0).unwrap().as_str())
                ))
            }
        }

        Ok(self.core_bpe._encode_native(text, &allowed_special_set).0)
    }

    /// Encodes a list of strings into tokens, in parallel.
    ///
    /// See `encode` for more details on `allowed_special` and `disallowed_special`.
    pub fn encode_batch(&self,
                        texts: Vec<&str>,
                        allowed_special: AllowedSpecial,
                        disallowed_special: DisallowedSpecial
    ) -> Result<Vec<Vec<usize>>> {
        let data: Vec<Result<Vec<usize>>> = texts.par_iter()
            .map(|&txt| self.encode(txt, allowed_special.clone(), disallowed_special.clone()))
            .collect();

        let mut res = Vec::new();
        for item in data {
            res.push(item?);
        }
        Ok(res)
    }

    /// Encodes a string into stable tokens and possible completion sequences.
    /// Note that the stable tokens will only represent a substring of `text`.
    /// See `encode` for more details on `allowed_special` and `disallowed_special`.
    /// This API should itself be considered unstable.
    pub fn encode_with_unstable(&self,
                                text: &str,
                                allowed_special: AllowedSpecial,
                                disallowed_special: DisallowedSpecial
    ) -> Result<(Vec<usize>, Vec<Vec<usize>>)> {

        let allowed_special_set = match allowed_special {
            AllowedSpecial::All => self.special_tokens_set(),
            AllowedSpecial::Allowed(allowed) => allowed,
        };
        let disallowed_special_set = match disallowed_special {
            DisallowedSpecial::All =>
                self.special_tokens_set()
                    .difference(&allowed_special_set)
                    .copied()
                    .collect(),
            DisallowedSpecial::Disallowed(disallowed) => disallowed,
        };

        if !disallowed_special_set.is_empty() {
            let re = special_token_regex(disallowed_special_set)?;
            if let Ok(Some(cap)) = re.captures(text) {
                return Err(EncodeError::SpecialTokenError(
                    String::from(cap.get(0).unwrap().as_str())
                ))
            }
        }

        let (tokens, completions) = self
            .core_bpe
            ._encode_unstable_native(text, &allowed_special_set);
        let completions = completions.into_iter().map(|seq| seq).collect();
        Ok((tokens, completions))
    }

    /// Encodes text corresponding to a single token to its token value.
    ///
    /// NOTE: this will encode all special tokens.
    pub fn encode_single_token(&self, piece: &[u8]) -> Result<usize> {
        if let Some(token) = self.core_bpe.encoder.get(piece).copied() {
            return Ok(token);
        }
        if let Ok(piece_str) = std::str::from_utf8(piece) {
            if let Some(token) = self.core_bpe.special_tokens_encoder.get(piece_str).copied() {
                return Ok(token);
            }
        }
        Err(EncodeError::TokenEncodeError(piece.to_owned()))
    }
}

/// Public interfaces for decoding
impl Encoding {

    /// Decodes a list of tokens into bytes.
    pub fn decode_bytes(&self, tokens: &Vec<usize>) -> Vec<u8> {
        self.core_bpe._decode_native(tokens)
    }

    /// Decodes a list of tokens into a string.
    ///
    /// WARNING: decoded bytes are not guaranteed to be valid UTF-8.
    /// You can control this behaviour using the `mode` parameter.
    /// `Strict` mode does validity check and returns Err if provided bytes are not UTF-8
    /// `Replace` mode replaces invalid UTF-8 sequences with U+FFFD
    ///
    pub fn decode(&self, tokens: &Vec<usize>, mode: DecodeMode) -> Result<String> {
        let bytes = self.decode_bytes(tokens);
        match mode {
            DecodeMode::Strict => String::from_utf8(bytes)
                .map_err(|e| EncodeError::ConvertStringError(e)),
            DecodeMode::Replace => Ok(String::from_utf8_lossy(&bytes).to_string()),
        }
    }

    /// Decodes a token into bytes.
    /// NOTE: this will decode all special tokens.
    pub fn decode_single_token_bytes(&self, token: usize) -> Result<Vec<u8>> {
        if let Some(bytes) = self.core_bpe.decoder.get(&token) {
            return Ok(bytes.to_vec());
        }
        if let Some(bytes) = self.core_bpe.special_tokens_decoder.get(&token) {
            return Ok(bytes.to_vec());
        }
        Err(EncodeError::TokenNotFoundError(token))
    }

    /// Decodes a list of tokens into a list of bytes.
    /// Useful for visualising tokenisation.
    pub fn decode_tokens_bytes(&self, tokens: &Vec<usize>) -> Result<Vec<Vec<u8>>> {
        let data: Vec<Result<Vec<u8>>> = tokens.par_iter()
            .map(|&token| self.decode_single_token_bytes(token))
            .collect();

        let mut res = Vec::new();
        for item in data {
            res.push(item?);
        }
        Ok(res)
    }
}

/// Miscellaneous interfaces
impl Encoding {
    /// Returns the list of all token byte values.
    pub fn token_byte_values(&self) -> Vec<Vec<u8>> {
        self.core_bpe
            .sorted_token_bytes
            .iter()
            .map(|x| x.to_vec())
            .collect()
    }

    pub fn eot_token(&self) -> Option<usize> {
        self.special_tokens
            .get("<|endoftext|>")
            .copied()
    }

    /// For backwards compatibility. Prefer to use `enc.max_token_value + 1`.
    pub fn n_vocab(&self) -> usize {
        self.max_token_value + 1
    }

    // TODO: lazy evaluation
    pub fn special_tokens_set(&self) -> HashSet<&str> {
        HashSet::from_iter(self
            .special_tokens
            .keys()
            .map(|k| k.as_str()))
    }
}

// TODO: LRU cache
fn special_token_regex(tokens: HashSet<&str>) -> Result<Regex> {
    let inner: Vec<_> = tokens.iter()
        .map(|&t| regex::escape(t))
        .collect();
    let re = Regex::new(format!("({})", inner.join("|")).as_str())?;
    Ok(re)
}

fn convert_to_fx_hashmap<K, V>(origin: &HashMap<K, V>)
    -> FxHashMap<K, V>
    where K: Hash + Eq + PartialEq + Clone, V: Clone {

    let mut res: FxHashMap<K, V> = FxHashMap::default();
    origin.iter().for_each(|(k, v)| _ = res.insert(k.clone(), v.clone()));
    res
}
