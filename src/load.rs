use std::{env, fs, io};
use std::collections::HashMap;
use std::ops::Add;
use std::path::{Path, PathBuf};
use rayon::prelude::*;
use sha2::{Sha256, Digest};
use uuid::Uuid;
use base64ct::{Base64, Encoding};
use bstr::ByteSlice;
use serde_json::{Result as JResult, Value, Map};
use crate::core::Result;

const TIKTOKEN_CACHE_DIR: &str = "TIKTOKEN_CACHE_DIR";
const DATA_GYM_CACHE_DIR: &str = "DATA_GYM_CACHE_DIR";
const DATA_GYM_TMP_DIR: &str = "data-gym-cache";

/// `blobpath` should have format like `https://<account>.blob.core.windows.net/<container>/`
///
/// TODO: support more format of blob storage path. For example,
/// Google Cloud Storage paths (gs://<bucket>)
/// Azure Blob Storage paths (az://<account>/<container>)
fn read_file_remote(blob_path: &str) -> Result<String> {
    let res = reqwest::blocking::get(blob_path)?.text()?;
    Ok(res)
}

fn read_file_cached(blob_path: &str) -> Result<String> {
    let cache_dir = env::var(TIKTOKEN_CACHE_DIR)
        .or(env::var(DATA_GYM_CACHE_DIR))
        .unwrap_or(
            env::temp_dir()
                .join(DATA_GYM_TMP_DIR)
                .to_string_lossy()
                .to_string()
        );

    if cache_dir.is_empty() {
        // disable caching
        return read_file_remote(blob_path)
    }

    let cache_key = Sha256::digest(blob_path);
    let cache_path = Path::new(&cache_dir)
        .join(String::from_utf8(cache_key.to_vec()).unwrap());
    if cache_path.exists() {
        let res = fs::read_to_string(&cache_path)?;
        return Ok(res)
    }

    let contents = read_file_remote(blob_path)?;
    fs::create_dir_all(&cache_dir)?;
    // cache_key is valid sha256 digest, so always valid to be a string.
    let mut tmp_file = String::from_utf8(cache_key.to_vec()).unwrap();
    let tmp_file = tmp_file + "." + Uuid::new_v4().to_string().as_str() + ".tmp";
    let tmp_cache_path = Path::new(&cache_dir).join(tmp_file);

    fs::write(&tmp_cache_path, &contents)?;
    fs::rename(&tmp_cache_path, &cache_path)?;

    Ok(contents)
}

/// panic if there's one line that either `key` part is not base64 encoded,
/// or `value` part is not a number.
pub fn load_tiktoken_bpe(tiktoken_bpe_file: &str) -> HashMap<Vec<u8>, usize> {
    let contents = read_file_cached(tiktoken_bpe_file).unwrap_or_default();
    contents.lines()
        .map(|line| line.split_once(" "))
        .filter(|item| item.is_some())
        .map(|item| {
            let (b64, num) = item.unwrap();
            let key = Vec::from(Base64::decode_vec(b64).unwrap());
            let val: usize = num.parse().unwrap();
            (key, val)
        })
        .collect()
}

/// Handle extended ascii (https://en.wikipedia.org/wiki/Extended_ASCII)
/// Assume ISO/IEC 8859-1 (https://en.wikipedia.org/wiki/ISO/IEC_8859-1)
/// non-whitespace printable character range:
/// [0x21-0x7E], [0xA1-0xAD), (0xAD-0xFF]
pub fn data_gym_to_mergeable_bpe_ranks(
    vocab_bpe_file: &str,
    encoder_json_file: &str
) -> HashMap<Vec<u8>, usize> {
    let mut rank_to_intbyte: Vec<u8> = vec![];
    rank_to_intbyte.extend(0x21..=0x7E);
    rank_to_intbyte.extend(0xA1..0xAD);
    rank_to_intbyte.extend(0xAE..=0xFF);

    let mut data_gym_byte_to_byte: HashMap<char, u8> = rank_to_intbyte
        .iter()
        .map(|&b| (char::from(b), b))
        .collect();
    let mut n = 0u32;
    for b in 0..=255 {
        if !rank_to_intbyte.contains(&b) {
            rank_to_intbyte.push(b);
            data_gym_byte_to_byte.insert(char::from_u32(256+n).unwrap(), b);
            n += 1;
        }
    }
    assert_eq!(rank_to_intbyte.len(), 256);

    // add the single byte tokens
    let mut bpe_ranks: HashMap<Vec<u8>, usize> = rank_to_intbyte
        .into_iter()
        .enumerate()
        .map(|(i, b)| (vec![b], i))
        .collect();

    // vocab_bpe contains the merges along with associated ranks
    let vocab_bpe_contents = read_file_cached(vocab_bpe_file).unwrap_or_default();
    let bpe_merges: Vec<(&str, &str)> = vocab_bpe_contents
        .lines()
        .skip(1)
        .map(|line| line.split_once(" "))
        .filter(|item| item.is_some())
        .map(|item| item.unwrap())
        .collect();

    let mut n = bpe_ranks.len();
    for (first, second) in bpe_merges {
        let mut key = decode_data_gym(first, &data_gym_byte_to_byte);
        key.extend(decode_data_gym(second, &data_gym_byte_to_byte));
        bpe_ranks.insert(key, n);
        n += 1
    }

    /// check that the encoder file matches the merges file
    /// this sanity check is important since tiktoken assumes that ranks are ordered the same
    /// as merge priority
    let content = read_file_cached(encoder_json_file)
        .unwrap_or("{}".to_string());
    let encoder_json: Value = serde_json::from_str(&content)
        .unwrap_or(Value::Object(Map::default()));
    let mut encoder_json_loaded: HashMap<Vec<u8>, usize> = encoder_json
        .as_object()
        .unwrap()
        .iter()
        .map(|(key, val)| {
            (decode_data_gym(key, &data_gym_byte_to_byte), val.as_u64().unwrap() as usize)
        })
        .collect();
    encoder_json_loaded.remove(b"<|endoftext|>".as_bytes());
    encoder_json_loaded.remove(b"<|endoftext|>".as_bytes());


    // TODO: assert bpe_ranks == encoder_json_loaded

    bpe_ranks
}

fn decode_data_gym(value: &str, dict: &HashMap<char, u8>) -> Vec<u8> {
    value.chars().map(|c| dict.get(&c).copied().unwrap()).collect()
}
