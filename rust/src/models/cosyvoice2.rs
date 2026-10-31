use anyhow::{anyhow, bail, Result};
use serde_json::Value;
use tokenizers::Tokenizer;

use super::{meta_ids, tokenizer, GenerateInput, Prompt, PromptBuilder};

/// CosyVoice2 with the default (LJSpeech) reference prompt: `[sos] ref_text text [task] ref_speech`.
pub struct CosyVoice2 {
    tokenizer: Tokenizer,
    sos: i32,
    task_id: i32,
    ref_text: Vec<i32>,
    ref_speech: Vec<i32>,
    kinds: [i32; 3],
}

impl CosyVoice2 {
    pub fn new(meta: &Value) -> Result<Self> {
        let int = |v: &Value, k: &str| v[k].as_i64().map(|x| x as i32).ok_or_else(|| anyhow!("meta.{k} missing"));
        let kinds =
            [int(&meta["row_kinds"], "special")?, int(&meta["row_kinds"], "text")?, int(&meta["row_kinds"], "speech")?];
        Ok(Self {
            tokenizer: tokenizer(meta, "tokenizer_json")?,
            sos: int(meta, "sos")?,
            task_id: int(meta, "task_id")?,
            ref_text: meta_ids(meta, "ref_text_ids"),
            ref_speech: meta_ids(meta, "ref_speech_tokens"),
            kinds,
        })
    }
}

impl PromptBuilder for CosyVoice2 {
    fn preprocess(&self, input: &GenerateInput) -> Result<Prompt> {
        if input.audio_path.is_some() {
            bail!("audio_path is not supported yet for this model");
        }
        let text = text_normalize(&input.text)?;
        let enc = self.tokenizer.encode(text, true).map_err(|e| anyhow!("tokenizer: {e}"))?;
        let [special, txt, speech] = self.kinds;
        let mut tokens = vec![self.sos];
        tokens.extend_from_slice(&self.ref_text);
        tokens.extend(enc.get_ids().iter().map(|&x| x as i32));
        let mut mask = vec![special];
        mask.resize(tokens.len(), txt);
        tokens.push(self.task_id);
        mask.push(special);
        tokens.extend_from_slice(&self.ref_speech);
        mask.resize(tokens.len(), speech);
        Ok(Prompt { tokens, mask: Some(mask), ..Default::default() })
    }
}

/// `CosyVoice2Model._text_normalize(text, split=False)`: strip and spell out digit runs.
pub(crate) fn text_normalize(text: &str) -> Result<String> {
    if (text.contains("<|") && text.contains("|>")) || text.is_empty() {
        return Ok(text.to_string());
    }
    let text = text.trim();
    let mut out = String::with_capacity(text.len() * 2);
    let mut digits = String::new();
    for c in text.chars() {
        if c.is_ascii_digit() {
            digits.push(c);
            continue;
        }
        if !digits.is_empty() {
            out.push_str(&number_to_words(&digits)?);
            digits.clear();
        }
        out.push(c);
    }
    if !digits.is_empty() {
        out.push_str(&number_to_words(&digits)?);
    }
    Ok(out)
}

const UNIT: [&str; 10] = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"];
const TEEN: [&str; 10] =
    ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"];
const TEN: [&str; 10] = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];
const MILL: [&str; 12] = [
    " ",
    " thousand",
    " million",
    " billion",
    " trillion",
    " quadrillion",
    " quintillion",
    " sextillion",
    " septillion",
    " octillion",
    " nonillion",
    " decillion",
];

fn mill(i: usize) -> Result<&'static str> {
    MILL.get(i).copied().ok_or_else(|| anyhow!("number out of range"))
}

fn tenfn(t: usize, u: usize, m: usize) -> Result<String> {
    if t != 1 {
        let hyphen = if t != 0 && u != 0 { "-" } else { "" };
        Ok(format!("{}{hyphen}{}{}", TEN[t], UNIT[u], mill(m)?))
    } else {
        Ok(format!("{}{}", TEEN[u], mill(m)?))
    }
}

fn hundfn(h: usize, t: usize, u: usize, m: usize) -> Result<String> {
    if h != 0 {
        let and = if t != 0 || u != 0 { " and " } else { "" };
        Ok(format!("{} hundred{and}{}{}, ", UNIT[h], tenfn(t, u, 0)?, mill(m)?))
    } else if t != 0 || u != 0 {
        Ok(format!("{}{}, ", tenfn(t, u, 0)?, mill(m)?))
    } else {
        Ok(String::new())
    }
}

/// Port of `inflect.engine().number_to_words(digits)` (default options) for an ASCII digit run.
pub fn number_to_words(num: &str) -> Result<String> {
    let trimmed = num.trim_start_matches('0');
    if trimmed.is_empty() {
        return Ok("zero".into());
    }
    if trimmed == "1" {
        return Ok("one".into());
    }
    let d: Vec<usize> = trimmed.bytes().map(|b| (b - b'0') as usize).collect();
    let mut rest = d.len();
    let mut m = 0;
    let mut s = String::new();
    while rest >= 3 {
        s = hundfn(d[rest - 3], d[rest - 2], d[rest - 1], m)? + &s;
        rest -= 3;
        m += 1;
    }
    if rest == 2 {
        s = tenfn(d[0], d[1], m)? + ", " + &s;
    } else if rest == 1 {
        s = format!("{}{}, ", UNIT[d[0]], mill(m)?) + &s;
    }
    // Post-processing of `number_to_words._handle_chunk`.
    if let Some(x) = s.strip_suffix(", ") {
        s = x.to_string();
    }
    s = collapse_ws_before_comma(&s);
    let t = s.trim_end();
    if t.len() < s.len() {
        if let Some(ws) = t.rfind(char::is_whitespace) {
            let word_start = ws + t[ws..].chars().next().unwrap().len_utf8();
            if t[..word_start].ends_with(", ") {
                s = format!("{} and {}", &t[..word_start - 2], &t[word_start..]);
            }
        }
    }
    Ok(s.split_whitespace().collect::<Vec<_>>().join(" "))
}

/// `re.sub(r"\s+,", ",", s)`.
fn collapse_ws_before_comma(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut pending = String::new();
    for c in s.chars() {
        if c.is_whitespace() {
            pending.push(c);
        } else {
            if c != ',' {
                out.push_str(&pending);
            }
            pending.clear();
            out.push(c);
        }
    }
    out.push_str(&pending);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn number_to_words_matches_inflect() {
        for (n, w) in [
            ("0", "zero"),
            ("7", "seven"),
            ("10", "ten"),
            ("13", "thirteen"),
            ("21", "twenty-one"),
            ("100", "one hundred"),
            ("101", "one hundred and one"),
            ("110", "one hundred and ten"),
            ("999", "nine hundred and ninety-nine"),
            ("1000", "one thousand"),
            ("1001", "one thousand and one"),
            ("1010", "one thousand and ten"),
            ("1100", "one thousand, one hundred"),
            ("2024", "two thousand and twenty-four"),
            ("12345", "twelve thousand, three hundred and forty-five"),
            ("100000", "one hundred thousand"),
            ("1000000", "one million"),
            ("1000001", "one million and one"),
            ("007", "seven"),
            ("0012", "twelve"),
            (
                "1234567890",
                "one billion, two hundred and thirty-four million, five hundred and sixty-seven thousand, eight hundred and ninety",
            ),
            ("10000000000000000", "ten quadrillion"),
            ("40", "forty"),
            ("1000000000000", "one trillion"),
        ] {
            assert_eq!(number_to_words(n).unwrap(), w, "{n}");
        }
    }

    #[test]
    fn normalize_spells_numbers() {
        assert_eq!(text_normalize("  I have 3 cats and 21 dogs ").unwrap(), "I have three cats and twenty-one dogs");
        assert_eq!(text_normalize("<|en|>keep 5").unwrap(), "<|en|>keep 5");
    }
}
