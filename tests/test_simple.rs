use tiktoken_rust as tt;

#[test]
fn test_get_encoding() -> tt::Result<()> {
    let enc = tt::get_encoding("gpt2")?;
    assert_eq!(
        enc.encode(
            "hello world",
            tt::AllowedSpecial::All,
            tt::DisallowedSpecial::All
        )?,
        vec![31373, 995]
    );
    assert_eq!(
        enc.decode(&vec![31373, 995], tt::DecodeMode::Strict)?,
        String::from("hello world")
    );
    assert_eq!(
        enc.encode(
            "hello <|endoftext|>",
            tt::AllowedSpecial::All,
            tt::DisallowedSpecial::All
        )?,
        vec![31373, 220, 50256]
    );

    let enc = tt::get_encoding("cl100k_base")?;
    assert_eq!(
        enc.encode(
            "hello world",
            tt::AllowedSpecial::All,
            tt::DisallowedSpecial::All
        )?,
        vec![15339, 1917]
    );
    assert_eq!(
        enc.decode(&vec![15339, 1917], tt::DecodeMode::Strict)?,
        String::from("hello world")
    );
    assert_eq!(
        enc.encode(
            "hello <|endoftext|>",
            tt::AllowedSpecial::All,
            tt::DisallowedSpecial::All
        )?,
        vec![15339, 220, 100257]
    );

    for enc_name in tt::list_encoding_names() {
        let enc = tt::get_encoding(enc_name)?;
        for token in 1..10_000 {
            assert_eq!(
                enc.encode_single_token(&enc.decode_single_token_bytes(token)?)?,
                token
            )
        }
    }

    Ok(())
}

#[test]
fn test_encoding_for_model() -> tt::Result<()> {
    let enc = tt::encoding_for_model("gpt2")?;
    assert_eq!(enc.name(), "gpt2");
    let enc = tt::encoding_for_model("text-davinci-003")?;
    assert_eq!(enc.name(), "p50k_base");
    let enc = tt::encoding_for_model("text-davinci-edit-001")?;
    assert_eq!(enc.name(), "p50k_edit");
    let enc = tt::encoding_for_model("gpt-3.5-turbo-0301")?;
    assert_eq!(enc.name(), "cl100k_base");

    Ok(())
}
