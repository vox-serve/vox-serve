import logging
import os
import re
import textwrap
from functools import cached_property
from typing import Any

import torch
from ko_speech_tools import hangul_romanize
from num2words import num2words
from tokenizers import Tokenizer


logger = logging.getLogger(__name__)

_whitespace_re = re.compile(r"\s+")


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text).strip()


def lowercase(text):
    return text.lower()


def get_spacy_lang(lang):
    try:
        from spacy.lang.ar import Arabic
        from spacy.lang.en import English
        from spacy.lang.es import Spanish
        from spacy.lang.hi import Hindi
        from spacy.lang.ja import Japanese
        from spacy.lang.zh import Chinese
    except ImportError as e:
        raise ImportError("enable_text_splitting=True requires Spacy: pip install spacy[ja]") from e
    """Return Spacy language used for sentence splitting."""
    if lang == "zh":
        return Chinese()
    elif lang == "ja":
        return Japanese()
    elif lang == "ar":
        return Arabic()
    elif lang == "es":
        return Spanish()
    elif lang == "hi":
        return Hindi()
    else:
        # For most languages, English does the job
        return English()


def split_sentence(text, lang, text_split_length=250):
    """Preprocess the input text"""
    text_splits = []
    if text_split_length is not None and len(text) >= text_split_length:
        text_splits.append("")
        nlp = get_spacy_lang(lang)
        nlp.add_pipe("sentencizer")
        doc = nlp(text)
        for sentence in doc.sents:
            if len(text_splits[-1]) + len(str(sentence)) <= text_split_length:
                # if the last sentence + the current sentence is less than the text_split_length
                # then add the current sentence to the last sentence
                text_splits[-1] += " " + str(sentence)
                text_splits[-1] = text_splits[-1].lstrip()
            elif len(str(sentence)) > text_split_length:
                # if the current sentence is greater than the text_split_length
                for line in textwrap.wrap(
                    str(sentence),
                    width=text_split_length,
                    drop_whitespace=True,
                    break_on_hyphens=False,
                    tabsize=1,
                ):
                    text_splits.append(str(line))
            else:
                text_splits.append(str(sentence))

        if len(text_splits) > 1:
            if text_splits[0] == "":
                del text_splits[0]
    else:
        text_splits = [text.lstrip()]

    return text_splits


# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = {
    "en": [
        (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
        for x in [
            ("mrs", "misess"),
            ("mr", "mister"),
            ("dr", "doctor"),
            ("st", "saint"),
            ("co", "company"),
            ("jr", "junior"),
            ("maj", "major"),
            ("gen", "general"),
            ("drs", "doctors"),
            ("rev", "reverend"),
            ("lt", "lieutenant"),
            ("hon", "honorable"),
            ("sgt", "sergeant"),
            ("capt", "captain"),
            ("esq", "esquire"),
            ("ltd", "limited"),
            ("col", "colonel"),
            ("ft", "fort"),
        ]
    ],
    "es": [
        (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
        for x in [
            ("sra", "señora"),
            ("sr", "señor"),
            ("dr", "doctor"),
            ("dra", "doctora"),
            ("st", "santo"),
            ("co", "compañía"),
            ("jr", "junior"),
            ("ltd", "limitada"),
        ]
    ],
    "fr": [
        (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
        for x in [
            ("mme", "madame"),
            ("mr", "monsieur"),
            ("dr", "docteur"),
            ("st", "saint"),
            ("co", "compagnie"),
            ("jr", "junior"),
            ("ltd", "limitée"),
        ]
    ],
    "de": [
        (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
        for x in [
            ("fr", "frau"),
            ("dr", "doktor"),
            ("st", "sankt"),
            ("co", "firma"),
            ("jr", "junior"),
        ]
    ],
    "pt": [
        (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
        for x in [
            ("sra", "senhora"),
            ("sr", "senhor"),
            ("dr", "doutor"),
            ("dra", "doutora"),
            ("st", "santo"),
            ("co", "companhia"),
            ("jr", "júnior"),
            ("ltd", "limitada"),
        ]
    ],
    "it": [
        (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
        for x in [
            # ("sig.ra", "signora"),
            ("sig", "signore"),
            ("dr", "dottore"),
            ("st", "santo"),
            ("co", "compagnia"),
            ("jr", "junior"),
            ("ltd", "limitata"),
        ]
    ],
    "pl": [
        (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
        for x in [
            ("p", "pani"),
            ("m", "pan"),
            ("dr", "doktor"),
            ("sw", "święty"),
            ("jr", "junior"),
        ]
    ],
    "ar": [
        (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
        for x in [
            # There are not many common abbreviations in Arabic as in English.
        ]
    ],
    "zh": [
        (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
        for x in [
            # Chinese doesn't typically use abbreviations in the same way as Latin-based scripts.
        ]
    ],
    "cs": [
        (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
        for x in [
            ("dr", "doktor"),  # doctor
            ("ing", "inženýr"),  # engineer
            ("p", "pan"),  # Could also map to pani for woman but no easy way to do it
            # Other abbreviations would be specialized and not as common.
        ]
    ],
    "ru": [
        (re.compile(f"\\b{x[0]}\\b", re.IGNORECASE), x[1])
        for x in [
            ("г-жа", "госпожа"),  # Mrs.
            ("г-н", "господин"),  # Mr.
            ("д-р", "доктор"),  # doctor
            # Other abbreviations are less common or specialized.
        ]
    ],
    "nl": [
        (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
        for x in [
            ("dhr", "de heer"),  # Mr.
            ("mevr", "mevrouw"),  # Mrs.
            ("dr", "dokter"),  # doctor
            ("jhr", "jonkheer"),  # young lord or nobleman
            # Dutch uses more abbreviations, but these are the most common ones.
        ]
    ],
    "tr": [
        (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
        for x in [
            ("b", "bay"),  # Mr.
            ("byk", "büyük"),  # büyük
            ("dr", "doktor"),  # doctor
            # Add other Turkish abbreviations here if needed.
        ]
    ],
    "hu": [
        (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
        for x in [
            ("dr", "doktor"),  # doctor
            ("b", "bácsi"),  # Mr.
            ("nőv", "nővér"),  # nurse
            # Add other Hungarian abbreviations here if needed.
        ]
    ],
    "ko": [
        (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
        for x in [
            # Korean doesn't typically use abbreviations in the same way as Latin-based scripts.
        ]
    ],
    "hi": [
        (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
        for x in [
            # Hindi doesn't typically use abbreviations in the same way as Latin-based scripts.
        ]
    ],
}


def expand_abbreviations_multilingual(text, lang="en"):
    for regex, replacement in _abbreviations[lang]:
        text = re.sub(regex, replacement, text)
    return text


_symbols_multilingual = {
    "en": [
        (re.compile(rf"{re.escape(x[0])}", re.IGNORECASE), x[1])
        for x in [
            ("&", " and "),
            ("@", " at "),
            ("%", " percent "),
            ("#", " hash "),
            ("$", " dollar "),
            ("£", " pound "),
            ("°", " degree "),
        ]
    ],
    "es": [
        (re.compile(rf"{re.escape(x[0])}", re.IGNORECASE), x[1])
        for x in [
            ("&", " y "),
            ("@", " arroba "),
            ("%", " por ciento "),
            ("#", " numeral "),
            ("$", " dolar "),
            ("£", " libra "),
            ("°", " grados "),
        ]
    ],
    "fr": [
        (re.compile(rf"{re.escape(x[0])}", re.IGNORECASE), x[1])
        for x in [
            ("&", " et "),
            ("@", " arobase "),
            ("%", " pour cent "),
            ("#", " dièse "),
            ("$", " dollar "),
            ("£", " livre "),
            ("°", " degrés "),
        ]
    ],
    "de": [
        (re.compile(rf"{re.escape(x[0])}", re.IGNORECASE), x[1])
        for x in [
            ("&", " und "),
            ("@", " at "),
            ("%", " prozent "),
            ("#", " raute "),
            ("$", " dollar "),
            ("£", " pfund "),
            ("°", " grad "),
        ]
    ],
    "pt": [
        (re.compile(rf"{re.escape(x[0])}", re.IGNORECASE), x[1])
        for x in [
            ("&", " e "),
            ("@", " arroba "),
            ("%", " por cento "),
            ("#", " cardinal "),
            ("$", " dólar "),
            ("£", " libra "),
            ("°", " graus "),
        ]
    ],
    "it": [
        (re.compile(rf"{re.escape(x[0])}", re.IGNORECASE), x[1])
        for x in [
            ("&", " e "),
            ("@", " chiocciola "),
            ("%", " per cento "),
            ("#", " cancelletto "),
            ("$", " dollaro "),
            ("£", " sterlina "),
            ("°", " gradi "),
        ]
    ],
    "pl": [
        (re.compile(rf"{re.escape(x[0])}", re.IGNORECASE), x[1])
        for x in [
            ("&", " i "),
            ("@", " małpa "),
            ("%", " procent "),
            ("#", " krzyżyk "),
            ("$", " dolar "),
            ("£", " funt "),
            ("°", " stopnie "),
        ]
    ],
    "ar": [
        # Arabic
        (re.compile(rf"{re.escape(x[0])}", re.IGNORECASE), x[1])
        for x in [
            ("&", " و "),
            ("@", " على "),
            ("%", " في المئة "),
            ("#", " رقم "),
            ("$", " دولار "),
            ("£", " جنيه "),
            ("°", " درجة "),
        ]
    ],
    "zh": [
        # Chinese
        (re.compile(rf"{re.escape(x[0])}", re.IGNORECASE), x[1])
        for x in [
            ("&", " 和 "),
            ("@", " 在 "),
            ("%", " 百分之 "),
            ("#", " 号 "),
            ("$", " 美元 "),
            ("£", " 英镑 "),
            ("°", " 度 "),
        ]
    ],
    "cs": [
        # Czech
        (re.compile(rf"{re.escape(x[0])}", re.IGNORECASE), x[1])
        for x in [
            ("&", " a "),
            ("@", " na "),
            ("%", " procento "),
            ("#", " křížek "),
            ("$", " dolar "),
            ("£", " libra "),
            ("°", " stupně "),
        ]
    ],
    "ru": [
        # Russian
        (re.compile(rf"{re.escape(x[0])}", re.IGNORECASE), x[1])
        for x in [
            ("&", " и "),
            ("@", " собака "),
            ("%", " процентов "),
            ("#", " номер "),
            ("$", " доллар "),
            ("£", " фунт "),
            ("°", " градус "),
        ]
    ],
    "nl": [
        # Dutch
        (re.compile(rf"{re.escape(x[0])}", re.IGNORECASE), x[1])
        for x in [
            ("&", " en "),
            ("@", " bij "),
            ("%", " procent "),
            ("#", " hekje "),
            ("$", " dollar "),
            ("£", " pond "),
            ("°", " graden "),
        ]
    ],
    "tr": [
        (re.compile(rf"{re.escape(x[0])}", re.IGNORECASE), x[1])
        for x in [
            ("&", " ve "),
            ("@", " at "),
            ("%", " yüzde "),
            ("#", " diyez "),
            ("$", " dolar "),
            ("£", " sterlin "),
            ("°", " derece "),
        ]
    ],
    "hu": [
        (re.compile(rf"{re.escape(x[0])}", re.IGNORECASE), x[1])
        for x in [
            ("&", " és "),
            ("@", " kukac "),
            ("%", " százalék "),
            ("#", " kettőskereszt "),
            ("$", " dollár "),
            ("£", " font "),
            ("°", " fok "),
        ]
    ],
    "ko": [
        # Korean
        (re.compile(rf"{re.escape(x[0])}", re.IGNORECASE), x[1])
        for x in [
            ("&", " 그리고 "),
            ("@", " 에 "),
            ("%", " 퍼센트 "),
            ("#", " 번호 "),
            ("$", " 달러 "),
            ("£", " 파운드 "),
            ("°", " 도 "),
        ]
    ],
    "hi": [
        (re.compile(rf"{re.escape(x[0])}", re.IGNORECASE), x[1])
        for x in [
            ("&", " और "),
            ("@", " ऐट दी रेट "),
            ("%", " प्रतिशत "),
            ("#", " हैश "),
            ("$", " डॉलर "),
            ("£", " पाउंड "),
            ("°", " डिग्री "),
        ]
    ],
}


def expand_symbols_multilingual(text, lang="en"):
    for regex, replacement in _symbols_multilingual[lang]:
        text = re.sub(regex, replacement, text)
        text = text.replace("  ", " ")  # Ensure there are no double spaces
    return text.strip()


_ordinal_re = {
    "en": re.compile(r"([0-9]+)(st|nd|rd|th)"),
    "es": re.compile(r"([0-9]+)(º|ª|er|o|a|os|as)"),
    "fr": re.compile(r"([0-9]+)(º|ª|er|re|e|ème)"),
    "de": re.compile(r"([0-9]+)(st|nd|rd|th|º|ª|\.(?=\s|$))"),
    "pt": re.compile(r"([0-9]+)(º|ª|o|a|os|as)"),
    "it": re.compile(r"([0-9]+)(º|°|ª|o|a|i|e)"),
    "pl": re.compile(r"([0-9]+)(º|ª|st|nd|rd|th)"),
    "ar": re.compile(r"([0-9]+)(ون|ين|ث|ر|ى)"),
    "cs": re.compile(r"([0-9]+)\.(?=\s|$)"),  # In Czech, a dot is often used after the number to indicate ordinals.
    "ru": re.compile(r"([0-9]+)(-й|-я|-е|-ое|-ье|-го)"),
    "nl": re.compile(r"([0-9]+)(de|ste|e)"),
    "tr": re.compile(r"([0-9]+)(\.|inci|nci|uncu|üncü|\.)"),
    "hu": re.compile(r"([0-9]+)(\.|adik|edik|odik|edik|ödik|ödike|ik)"),
    "ko": re.compile(r"([0-9]+)(번째|번|차|째)"),
    "hi": re.compile(r"([0-9]+)(st|nd|rd|th)"),  # To check
}
_number_re = re.compile(r"[0-9]+")
_currency_re = {
    "USD": re.compile(r"((\$[0-9\.\,]*[0-9]+)|([0-9\.\,]*[0-9]+\$))"),
    "GBP": re.compile(r"((£[0-9\.\,]*[0-9]+)|([0-9\.\,]*[0-9]+£))"),
    "EUR": re.compile(r"(([0-9\.\,]*[0-9]+€)|((€[0-9\.\,]*[0-9]+)))"),
}

_comma_number_re = re.compile(r"\b\d{1,3}(,\d{3})*(\.\d+)?\b")
_dot_number_re = re.compile(r"\b\d{1,3}(.\d{3})*(\,\d+)?\b")
_decimal_number_re = re.compile(r"([0-9]+[.,][0-9]+)")


def _remove_commas(m):
    text = m.group(0)
    if "," in text:
        text = text.replace(",", "")
    return text


def _remove_dots(m):
    text = m.group(0)
    if "." in text:
        text = text.replace(".", "")
    return text


def _expand_decimal_point(m, lang="en"):
    amount = m.group(1).replace(",", ".")
    return num2words(float(amount), lang=lang)


def _expand_currency(m, lang="en", currency="USD"):
    amount = float(re.sub(r"[^\d.]", "", m.group(0).replace(",", ".")))
    full_amount = num2words(amount, to="currency", currency=currency, lang=lang)

    and_equivalents = {
        "en": ", ",
        "es": " con ",
        "fr": " et ",
        "de": " und ",
        "pt": " e ",
        "it": " e ",
        "pl": ", ",
        "cs": ", ",
        "ru": ", ",
        "nl": ", ",
        "ar": ", ",
        "tr": ", ",
        "hu": ", ",
        "ko": ", ",
        "hi": ", ",
    }

    if amount.is_integer():
        last_and = full_amount.rfind(and_equivalents[lang])
        if last_and != -1:
            full_amount = full_amount[:last_and]

    return full_amount


def _expand_ordinal(m, lang="en"):
    return num2words(int(m.group(1)), ordinal=True, lang=lang)


def _expand_number(m, lang="en"):
    return num2words(int(m.group(0)), lang=lang)


def expand_numbers_multilingual(text, lang="en"):
    if lang == "zh":
        text = zh_num2words()(text)
    else:
        if lang in ["en", "ru"]:
            text = re.sub(_comma_number_re, _remove_commas, text)
        else:
            text = re.sub(_dot_number_re, _remove_dots, text)
        try:
            text = re.sub(_currency_re["GBP"], lambda m: _expand_currency(m, lang, "GBP"), text)
            text = re.sub(_currency_re["USD"], lambda m: _expand_currency(m, lang, "USD"), text)
            text = re.sub(_currency_re["EUR"], lambda m: _expand_currency(m, lang, "EUR"), text)
        except:
            pass
        if lang != "tr":
            text = re.sub(_decimal_number_re, lambda m: _expand_decimal_point(m, lang), text)
        text = re.sub(_ordinal_re[lang], lambda m: _expand_ordinal(m, lang), text)
        text = re.sub(_number_re, lambda m: _expand_number(m, lang), text)
    return text


def multilingual_cleaners(text, lang):
    text = text.replace('"', "")
    if lang == "tr":
        text = text.replace("İ", "i")
        text = text.replace("Ö", "ö")
        text = text.replace("Ü", "ü")
    text = lowercase(text)
    text = expand_numbers_multilingual(text, lang)
    text = expand_abbreviations_multilingual(text, lang)
    text = expand_symbols_multilingual(text, lang=lang)
    text = collapse_whitespace(text)
    return text


def chinese_transliterate(text):
    try:
        import pypinyin
    except ImportError as e:
        raise ImportError("Chinese requires: pypinyin") from e
    return "".join(
        [p[0] for p in pypinyin.pinyin(text, style=pypinyin.Style.TONE3, heteronym=False, neutral_tone_with_five=True)]
    )


def japanese_cleaners(text, katsu):
    text = katsu.romaji(text)
    text = lowercase(text)
    return text


class VoiceBpeTokenizer:
    def __init__(self, vocab_file: str | os.PathLike[Any] | None = None):
        self.tokenizer = None
        if vocab_file is not None:
            self.tokenizer = Tokenizer.from_file(str(vocab_file))
        self.char_limits = {
            "en": 250,
            "de": 253,
            "fr": 273,
            "es": 239,
            "it": 213,
            "pt": 203,
            "pl": 224,
            "zh": 82,
            "ar": 166,
            "cs": 186,
            "ru": 182,
            "nl": 251,
            "tr": 226,
            "ja": 71,
            "hu": 224,
            "ko": 95,
            "hi": 150,
        }

    @cached_property
    def katsu(self):
        import cutlet

        return cutlet.Cutlet()

    def check_input_length(self, txt, lang):
        lang = lang.split("-")[0]  # remove the region
        limit = self.char_limits.get(lang, 250)
        if len(txt) > limit:
            logger.warning(
                "The text length exceeds the character limit of %d for language '%s', this might cause truncated audio: %s",
                limit,
                lang,
                txt[:50] + "...",
            )

    def preprocess_text(self, txt, lang):
        if lang in {"ar", "cs", "de", "en", "es", "fr", "hi", "hu", "it", "nl", "pl", "pt", "ru", "tr", "zh", "ko"}:
            txt = multilingual_cleaners(txt, lang)
            if lang == "zh":
                txt = chinese_transliterate(txt)
            if lang == "ko":
                txt = hangul_romanize(txt)
        elif lang == "ja":
            txt = japanese_cleaners(txt, self.katsu)
        else:
            raise NotImplementedError(f"Language '{lang}' is not supported.")
        return txt

    def encode(self, txt, lang):
        lang = lang.split("-")[0]  # remove the region
        self.check_input_length(txt, lang)
        txt = self.preprocess_text(txt, lang)
        lang = "zh-cn" if lang == "zh" else lang
        txt = f"[{lang}]{txt}"
        txt = txt.replace(" ", "[SPACE]")
        return self.tokenizer.encode(txt).ids

    def decode(self, seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()
        txt = self.tokenizer.decode(seq, skip_special_tokens=False).replace(" ", "")
        txt = txt.replace("[SPACE]", " ")
        txt = txt.replace("[STOP]", "")
        txt = txt.replace("[UNK]", "")
        return txt

    def __len__(self):
        return self.tokenizer.get_vocab_size()

    def get_number_tokens(self):
        return max(self.tokenizer.get_vocab().values()) + 1