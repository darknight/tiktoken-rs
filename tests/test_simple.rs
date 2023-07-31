use tiktoken_rust as tt;

#[test]
fn test_simple() -> tt::Result<()> {
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
fn test_simple_repeated() -> tt::Result<()> {
    let enc = tt::get_encoding("gpt2")?;
    assert_eq!(
        enc.encode("0", tt::AllowedSpecial::All, tt::DisallowedSpecial::All)?,
        [15]
    );
    assert_eq!(
        enc.encode("00", tt::AllowedSpecial::All, tt::DisallowedSpecial::All)?,
        [405]
    );
    assert_eq!(
        enc.encode("000", tt::AllowedSpecial::All, tt::DisallowedSpecial::All)?,
        [830]
    );
    assert_eq!(
        enc.encode("0000", tt::AllowedSpecial::All, tt::DisallowedSpecial::All)?,
        [2388]
    );
    assert_eq!(
        enc.encode("00000", tt::AllowedSpecial::All, tt::DisallowedSpecial::All)?,
        [20483]
    );
    assert_eq!(
        enc.encode(
            "000000",
            tt::AllowedSpecial::All,
            tt::DisallowedSpecial::All
        )?,
        [10535]
    );
    assert_eq!(
        enc.encode(
            "0000000",
            tt::AllowedSpecial::All,
            tt::DisallowedSpecial::All
        )?,
        [24598]
    );
    assert_eq!(
        enc.encode(
            "00000000",
            tt::AllowedSpecial::All,
            tt::DisallowedSpecial::All
        )?,
        [8269]
    );
    assert_eq!(
        enc.encode(
            "000000000",
            tt::AllowedSpecial::All,
            tt::DisallowedSpecial::All
        )?,
        [10535, 830]
    );
    assert_eq!(
        enc.encode(
            "0000000000",
            tt::AllowedSpecial::All,
            tt::DisallowedSpecial::All
        )?,
        [8269, 405]
    );
    assert_eq!(
        enc.encode(
            "00000000000",
            tt::AllowedSpecial::All,
            tt::DisallowedSpecial::All
        )?,
        [8269, 830]
    );
    assert_eq!(
        enc.encode(
            "000000000000",
            tt::AllowedSpecial::All,
            tt::DisallowedSpecial::All
        )?,
        [8269, 2388]
    );
    assert_eq!(
        enc.encode(
            "0000000000000",
            tt::AllowedSpecial::All,
            tt::DisallowedSpecial::All
        )?,
        [8269, 20483]
    );
    assert_eq!(
        enc.encode(
            "00000000000000",
            tt::AllowedSpecial::All,
            tt::DisallowedSpecial::All
        )?,
        [8269, 10535]
    );
    assert_eq!(
        enc.encode(
            "000000000000000",
            tt::AllowedSpecial::All,
            tt::DisallowedSpecial::All
        )?,
        [8269, 24598]
    );
    assert_eq!(
        enc.encode(
            "0000000000000000",
            tt::AllowedSpecial::All,
            tt::DisallowedSpecial::All
        )?,
        [25645]
    );
    assert_eq!(
        enc.encode(
            "00000000000000000",
            tt::AllowedSpecial::All,
            tt::DisallowedSpecial::All
        )?,
        [8269, 10535, 830]
    );

    Ok(())
}

#[test]
fn test_simple_regex() -> tt::Result<()> {
    let enc = tt::get_encoding("cl100k_base")?;
    assert_eq!(
        enc.encode("rer", tt::AllowedSpecial::All, tt::DisallowedSpecial::All)?,
        [38149]
    );
    assert_eq!(
        enc.encode("'rer", tt::AllowedSpecial::All, tt::DisallowedSpecial::All)?,
        [2351, 81]
    );
    assert_eq!(
        enc.encode(
            "today\n ",
            tt::AllowedSpecial::All,
            tt::DisallowedSpecial::All
        )?,
        [31213, 198, 220]
    );
    assert_eq!(
        enc.encode(
            "today\n \n",
            tt::AllowedSpecial::All,
            tt::DisallowedSpecial::All
        )?,
        [31213, 27907]
    );
    assert_eq!(
        enc.encode(
            "today\n  \n",
            tt::AllowedSpecial::All,
            tt::DisallowedSpecial::All
        )?,
        [31213, 14211]
    );

    Ok(())
}

#[test]
fn test_basic_encode() -> tt::Result<()> {
    let enc = tt::get_encoding("r50k_base")?;
    assert_eq!(
        enc.encode(
            "hello world",
            tt::AllowedSpecial::All,
            tt::DisallowedSpecial::All
        )?,
        [31373, 995]
    );
    let enc = tt::get_encoding("p50k_base")?;
    assert_eq!(
        enc.encode(
            "hello world",
            tt::AllowedSpecial::All,
            tt::DisallowedSpecial::All
        )?,
        [31373, 995]
    );
    let enc = tt::get_encoding("cl100k_base")?;
    assert_eq!(
        enc.encode(
            "hello world",
            tt::AllowedSpecial::All,
            tt::DisallowedSpecial::All
        )?,
        [15339, 1917]
    );
    // py: chr(133) = '\x85'
    let case = b" \x850";
    // FIXME: failed case
    // assert_eq!(
    //     enc.encode(case, tt::AllowedSpecial::All, tt::DisallowedSpecial::All)?,
    //     [220, 126, 227, 15]
    // );
    Ok(())
}

#[test]
fn test_encode_empty() -> tt::Result<()> {
    let enc = tt::get_encoding("r50k_base")?;
    assert_eq!(
        enc.encode("", tt::AllowedSpecial::All, tt::DisallowedSpecial::All)?,
        Vec::<usize>::new()
    );

    Ok(())
}

#[test]
fn test_encode_bytes() -> tt::Result<()> {
    let enc = tt::get_encoding("cl100k_base")?;
    assert_eq!(enc.encode_single_token(b" \xec\x8b\xa4\xed")?, 62085);

    Ok(())
}

#[test]
fn test_encode_surrogate_pairs() -> tt::Result<()> {
    let enc = tt::get_encoding("cl100k_base")?;
    assert_eq!(
        enc.encode("👍", tt::AllowedSpecial::All, tt::DisallowedSpecial::All)?,
        [9468, 239, 235]
    );
    assert_eq!(
        enc.encode(
            &String::from_utf16(&[0xd83d, 0xdc4d]).unwrap(),
            tt::AllowedSpecial::All,
            tt::DisallowedSpecial::All
        )?,
        [9468, 239, 235]
    );
    // lone surrogate just gets replaced
    // FIXME: failed case
    // assert_eq!(
    //     enc.encode(
    //         &String::from_utf16(&[0xd83d]).unwrap(),
    //         tt::AllowedSpecial::All,
    //         tt::DisallowedSpecial::All
    //     )?,
    //     enc.encode(
    //         "�",
    //         tt::AllowedSpecial::All,
    //         tt::DisallowedSpecial::All
    //     )?
    // );
    Ok(())
}

#[test]
fn test_special_token() -> tt::Result<()> {
    let enc = tt::get_encoding("cl100k_base")?;
    let eot = enc.encode_single_token("<|endoftext|>".as_bytes())?;
    assert_eq!(eot, enc.eot_token().unwrap());

    // TODO

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

#[test]
fn test_decode_with_offsets() -> tt::Result<()> {
    let enc = tt::get_encoding("gpt2")?;
    let (text, offsets) = enc.decode_with_offsets(&vec![31373, 995])?;
    assert_eq!(text, "hello world");
    assert_eq!(offsets, &[0, 5]);

    Ok(())
}

#[test]
fn test_basic_offsets() -> tt::Result<()> {
    // TODO
    Ok(())
}
