# tiktoken-rust

**STATUS**: Under development.

[tiktoken](https://github.com/openai/tiktoken) is a fast [BPE](https://en.wikipedia.org/wiki/Byte_pair_encoding) tokeniser for use with
OpenAI's models. It provides Python interface to interact with it.

This project is a fork of original repo, bring the capability to rust world.

```rust
use tiktoken_rust as tt;

let enc = tt::get_encoding("cl100k_base").unwrap();

assert_eq!(
    "hello world",
    enc.decode(&enc.encode_ordinary("hello world"), tt::DecodeMode::Strict).unwrap()
)
```

