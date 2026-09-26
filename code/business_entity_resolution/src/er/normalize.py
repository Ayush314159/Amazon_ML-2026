"""Country-agnostic normalization of business names and addresses.

Nothing here branches on the `country` value. The vocabulary tables below are
plain word-level canonicalizations (street-type words, legal-form words) that
apply to any record containing those words, whichever country it comes from.
"""
import re
import unicodedata

# Bump whenever normalization output changes; stores/indexes/experiments record it.
# v1: case/accent/punctuation/abbreviation folding. v2: + Indic->Latin transliteration.
VERSION = 2


def _latin_fold_table() -> dict:
    """Map accented Latin letters to their ASCII base; leave other scripts alone."""
    table = {}
    for cp in range(0x00C0, 0x0250):
        ch = chr(cp)
        base = "".join(c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c))
        if base != ch and base.isascii() and base:
            table[cp] = base
    table.update({
        ord("ß"): "ss", ord("æ"): "ae", ord("Æ"): "ae", ord("œ"): "oe", ord("Œ"): "oe",
        ord("ø"): "o", ord("Ø"): "o", ord("đ"): "d", ord("Đ"): "d", ord("ł"): "l", ord("Ł"): "l",
        # apostrophes are dropped so "Orelee's" -> "orelees"
        ord("'"): "", ord("’"): "", ord("‘"): "", ord("`"): "", ord("´"): "",
        # U+FFFD: upstream encoding corruption ("FR�EDOM"); treat as a lost character
        0xFFFD: "",
        ord("&"): " and ", ord("+"): " and ",
    })
    return table


_FOLD = str.maketrans(_latin_fold_table())
# Keep ASCII alphanumerics and the Indic blocks (Devanagari .. Sinhala, U+0900-U+0DFF),
# which include their combining vowel signs. Everything else separates tokens.
_NON_TOKEN = re.compile(r"[^0-9a-zऀ-෿]+")
_ORDINAL = re.compile(r"\d+(st|nd|rd|th)")
_OCR = str.maketrans({"0": "o", "1": "l", "3": "e", "5": "s", "6": "g", "8": "b"})

# Legal-form / filler words across the vocabularies seen in train and test
# (US, India, and the French forms that only appear in test). Used to build the
# "core name" key; they stay in the token stream for similarity scoring.
LEGAL_FORMS = frozenset("""
llc inc incorporated corp corporation co company cos ltd limited pvt private llp lp
plc pc pllc pa dba the and of
sarl sas sasu eurl sa sci snc selarl scop scm
""".split())

NAME_CANON = {
    "limited": "ltd", "private": "pvt", "corporation": "corp", "incorporated": "inc",
    "company": "co", "enterprise": "enterprises", "centre": "center",
}

ADDR_CANON = {
    "street": "st", "str": "st", "road": "rd", "avenue": "ave", "av": "ave", "avn": "ave",
    "drive": "dr", "lane": "ln", "court": "ct", "boulevard": "blvd", "bd": "blvd",
    "place": "pl", "circle": "cir", "highway": "hwy", "parkway": "pkwy", "trail": "trl",
    "terrace": "ter", "square": "sq", "suite": "ste", "apartment": "apt", "building": "bldg",
    "floor": "fl", "north": "n", "south": "s", "east": "e", "west": "w", "near": "nr",
    "opposite": "opp", "number": "no", "mount": "mt", "fort": "ft", "saint": "st",
    "post": "po", "office": "off", "sector": "sec", "colony": "col", "nagar": "ngr",
}


# ---- Indic -> Latin transliteration -----------------------------------------
# The nine Brahmic blocks U+0900..U+0DFF (Devanagari, Bengali, Gurmukhi, Gujarati,
# Oriya, Tamil, Telugu, Kannada, Malayalam) share the ISCII layout, so one table keyed
# by the offset within a 0x80 block covers all of them. It is a deterministic
# character mapping (no model, no external data); spelling is approximate
# ("सिस्टम्स" -> "sistams"), which the phonetic key below absorbs.
_I_VOWEL = {0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri", 0x0C: "li",
            0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au"}
_I_CONS = {0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh", 0x1C: "j",
           0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n", 0x24: "t",
           0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n", 0x2A: "p", 0x2B: "ph", 0x2C: "b",
           0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l",
           0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h", 0x58: "q", 0x59: "kh", 0x5A: "g",
           0x5B: "z", 0x5C: "r", 0x5D: "rh", 0x5E: "f", 0x5F: "y"}
_I_MATRA = {0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri", 0x45: "e",
            0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au", 0x57: "au"}
_I_SIGN = {0x01: "n", 0x02: "n", 0x03: "h"}  # candrabindu, anusvara, visarga (Tamil aytham)
_I_VIRAMA, _I_NUKTA = 0x4D, 0x3C


def _is_indic(ch: str) -> bool:
    return 0x0900 <= ord(ch) <= 0x0DFF


def transliterate_indic(tok: str) -> str:
    """Latin approximation of a token containing Brahmic-script letters."""
    out = []
    pending_a = False  # inherent vowel of the last consonant, dropped at word end (schwa deletion)
    chars = [c for c in tok if not (_is_indic(c) and (ord(c) - 0x0900) % 0x80 == _I_NUKTA)]
    for i, ch in enumerate(chars):
        if not _is_indic(ch):
            if pending_a:
                out.append("a")
                pending_a = False
            out.append(ch)
            continue
        off = (ord(ch) - 0x0900) % 0x80
        if off in _I_CONS:
            if pending_a:
                out.append("a")
            nxt = chars[i + 1] if i + 1 < len(chars) else ""
            if off == 0x2A and out and out[-1] == "h" and i > 0 and (ord(chars[i - 1]) - 0x0900) % 0x80 == 0x03:
                out[-1] = "f"  # Tamil aytham + pa spells "f"
            else:
                out.append(_I_CONS[off])
            nxt_off = (ord(nxt) - 0x0900) % 0x80 if nxt and _is_indic(nxt) else None
            pending_a = nxt_off not in _I_MATRA and nxt_off != _I_VIRAMA
        elif off in _I_MATRA:
            out.append(_I_MATRA[off])
            pending_a = False
        elif off in _I_VOWEL:
            if pending_a:
                out.append("a")
            out.append(_I_VOWEL[off])
            pending_a = False
        elif off in _I_SIGN:
            if pending_a:
                out.append("a")
            out.append(_I_SIGN[off])
            pending_a = False
        elif 0x66 <= off <= 0x6F:
            if pending_a:
                out.append("a")
            out.append(str(off - 0x66))
            pending_a = False
        else:  # virama and anything unmapped
            pending_a = False
    return "".join(out)


def tokenize(text: str) -> list:
    if not text:
        return []
    toks = _NON_TOKEN.sub(" ", text.lower().translate(_FOLD)).split()
    return [transliterate_indic(t) if any(_is_indic(c) for c in t) else t for t in toks]


# ---- phonetic key --------------------------------------------------------------
_SOFT_C = re.compile(r"c(?=[eiy])")
_SOFT_G = re.compile(r"g(?=[eiy])")
_PHON = str.maketrans({"b": "p", "d": "t", "q": "k", "c": "k", "g": "k", "z": "s", "v": "f", "w": "f"})
_DROP = re.compile(r"[aeiouyh]")
_REPEAT = re.compile(r"(.)\1+")


def phonetic_key(tok: str) -> str:
    """Coarse consonant skeleton shared by spelling variants and transliterations:
    'systems'/'sistams' -> 'stms', 'private'/'praivet' -> 'prft', 'apex'/'epeks' -> 'pks'.
    Returns '' for numbers, non-Latin leftovers, and skeletons shorter than 2."""
    if not tok.isascii() or not tok.isalpha():
        return ""
    s = tok.replace("tion", "shn").replace("sion", "shn").replace("ph", "f").replace("x", "ks").replace("ck", "k")
    s = _SOFT_G.sub("j", _SOFT_C.sub("s", s))
    s = _REPEAT.sub(r"\1", _DROP.sub("", s.translate(_PHON)))
    return s if len(s) >= 2 else ""


def _fix_ocr(tok: str) -> str:
    """Undo digit-for-letter substitutions inside mostly-alphabetic tokens ("t0rres", "6rand")."""
    n_digit = sum(c.isdigit() for c in tok)
    if n_digit == 0 or n_digit > 2 or _ORDINAL.fullmatch(tok):
        return tok
    n_alpha = sum("a" <= c <= "z" for c in tok)
    if n_alpha >= 3 and n_alpha > 2 * n_digit:
        return tok.translate(_OCR)
    return tok


def name_tokens(name: str) -> list:
    out = []
    for t in tokenize(name):
        if t in ("www", "http", "https"):
            continue
        t = _fix_ocr(t)
        out.append(NAME_CANON.get(t, t))
    return out


def addr_tokens(addr: str) -> list:
    out = []
    for t in tokenize(addr):
        if t.isdigit():
            t = t.lstrip("0") or "0"
        out.append(ADDR_CANON.get(t, t))
    return out


def normalize_name(name: str) -> str:
    return " ".join(name_tokens(name))


def normalize_addr(addr: str) -> str:
    return " ".join(addr_tokens(addr))


# ---- blocking keys, derived from the normalized strings --------------------

def name_core_key(name_norm: str) -> str:
    """Order-insensitive name key without legal-form words ("llc sorrells and hanson" == "sorrells hanson llc")."""
    toks = {t for t in name_norm.split() if t not in LEGAL_FORMS}
    return " ".join(sorted(toks))


def addr_set_key(addr_norm: str) -> str:
    """Order-insensitive address key ("oh columbus 5559 orville ave" == "5559 orville ave columbus oh")."""
    return " ".join(sorted(set(addr_norm.split())))


def house_numbers(addr_norm: str) -> list:
    return [t for t in addr_norm.split() if t.isdigit()]
