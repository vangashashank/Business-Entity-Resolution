"""
ML Challenge 2026 — Business Entity Resolution
Module: src/preprocessing.py

Performs text normalization and standardization on business records:
1. Lowercase conversion
2. Punctuation and symbol stripping (preserving Unicode letters and combining marks for Hindi/Devanagari, Telugu, French accents)
3. Whitespace collapsing
4. Legal-suffix standardization (e.g., 'private limited' -> 'pvt ltd', 'corporation' -> 'corp')
5. Address abbreviation standardization (e.g., 'rd' -> 'road', 'st' -> 'street')
   Note: Avoids colliding stop words (such as 'de', 'la' in French or 'in', 'or' in English).

Keeps both raw and normalized columns in the resulting DataFrame.
"""

import sys
import re
import unicodedata
import pandas as pd
from typing import Optional, List, Dict

if hasattr(sys.stdout, "reconfigure") and sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# -----------------------------------------------------------------------------
# 1. Unicode Punctuation and Symbol Table
# -----------------------------------------------------------------------------
# Replaces Punctuation (P*) and Symbols (S*) with space.
# Preserves Letters (L*), Combining Marks (M* e.g. Hindi/Telugu matras), and Numbers (N*).
_PUNCT_TRANSLATE = str.maketrans({
    chr(i): " " for i in range(65536)
    if unicodedata.category(chr(i)).startswith(("P", "S"))
})

_WHITESPACE_RE = re.compile(r"\s+", flags=re.UNICODE)
_URL_RE = re.compile(r"(https?://\S+|www\.\S+|\b\S+\.(com|org|net|in|co|us|fr|gov|edu)\b)", flags=re.IGNORECASE)
_PREFIX_NOISE_RE = re.compile(r"^[\#\-\*\.\:\s]+", flags=re.UNICODE)

# -----------------------------------------------------------------------------
# 2. Dictionaries for Legal Suffixes & Address Standardization
# -----------------------------------------------------------------------------

# Legal suffix mapping: map variations to canonical forms
LEGAL_SUFFIXES_ORDERED = [
    # Multi-word
    (r"\bprivate\s+limited\b", "pvt ltd"),
    (r"\bpvt\s+limited\b", "pvt ltd"),
    (r"\bprivate\s+ltd\b", "pvt ltd"),
    (r"\bpvtltd\b", "pvt ltd"),
    (r"\blimited\s+liability\s+company\b", "llc"),
    (r"\blimited\s+liability\s+partnership\b", "llp"),
    (r"\bprofessional\s+limited\s+liability\s+company\b", "pllc"),
    (r"\bprofessional\s+corporation\b", "pc"),
    (r"\bjoint\s+stock\s+company\b", "jsc"),
    (r"\bpublic\s+limited\s+company\b", "plc"),
    # French entity types (open set support)
    (r"\bsociete\s+anonyme\b", "sa"),
    (r"\bsociete\s+par\s+actions\s+simplifiee\b", "sas"),
    (r"\bsociete\s+a\s+responsabilite\s+limitee\b", "sarl"),
    (r"\bentreprise\s+unipersonnelle\s+a\s+responsabilite\s+limitee\b", "eurl"),
    (r"\bsociete\s+civile\s+immobiliere\b", "sci"),
    (r"\bsasu\b", "sas"),
    # Single-word
    (r"\bcorporation\b", "corp"),
    (r"\bincorporated\b", "inc"),
    (r"\bcompany\b", "co"),
    (r"\blimited\b", "ltd"),
    (r"\bholdings?\b", "holding"),
    (r"\benterprises?\b", "enterprise"),
    (r"\bservices?\b", "service"),
    (r"\bconsultanc(y|ies)\b", "consultancy"),
    (r"\bassociates?\b", "assoc"),
    (r"\bpartners?\b", "partner"),
    # Devanagari legal suffixes
    (r"प्राइवेट\s+लिमिटेड", "pvt ltd"),
    (r"प्रा\s*लि", "pvt ltd"),
    (r"प्राइवेट", "pvt"),
    (r"लिमिटेड", "ltd"),
]

# Address token expansions / standardizations
# Excludes stop words that collide across languages (e.g. 'in', 'or', 'as', 'de', 'la')
ADDRESS_ABBREVIATIONS = {
    # Thoroughfares & street types
    r"\bst\b": "street",
    r"\bstr\b": "street",
    r"\brd\b": "road",
    r"\bave\b": "avenue",
    r"\bav\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bbvd\b": "boulevard",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bct\b": "court",
    r"\bcir\b": "circle",
    r"\bcl\b": "close",
    r"\bpkwy\b": "parkway",
    r"\bhwy\b": "highway",
    r"\bexpy\b": "expressway",
    r"\bway\b": "way",
    r"\bpl\b": "place",
    r"\bsq\b": "square",
    r"\bter\b": "terrace",
    # Buildings & units
    r"\bste\b": "suite",
    r"\bapt\b": "apartment",
    r"\bfl\b": "floor",
    r"\bflr\b": "floor",
    r"\bbldg\b": "building",
    r"\bdept\b": "department",
    r"\brm\b": "room",
    r"\bno\b": "number",
    # Landmarks & location terms
    r"\bopp\b": "opposite",
    r"\bnr\b": "near",
    r"\bdist\b": "district",
    r"\btq\b": "taluk",
    r"\bh\s+no\b": "house number",
    r"\bplt\b": "plot",
    r"\bsec\b": "sector",
    r"\bext\b": "extension",
    r"\bmkt\b": "market",
    # French thoroughfare standardizations (open set support)
    r"\bbd\b": "boulevard",
    r"\ball\b": "allee",
    r"\bche\b": "chemin",
    r"\bimp\b": "impasse",
    r"\brt\b": "route",
    # Distinct Indian state abbreviations (unambiguous)
    r"\bmh\b": "maharashtra",
    r"\bmp\b": "madhya pradesh",
    r"\bup\b": "uttar pradesh",
    r"\btn\b": "tamil nadu",
    r"\bdl\b": "delhi",
    r"\bts\b": "telangana",
    r"\btg\b": "telangana",
    r"\bap\b": "andhra pradesh",
    r"\brj\b": "rajasthan",
    r"\bgj\b": "gujarat",
    r"\bwb\b": "west bengal",
    r"\bkl\b": "kerala",
    r"\bhr\b": "haryana",
    r"\bpb\b": "punjab",
    r"\bjh\b": "jharkhand",
    r"\bct\b": "chhattisgarh",
    r"\bcg\b": "chhattisgarh",
    r"\buk\b": "uttarakhand",
    r"\bua\b": "uttarakhand",
    r"\bhp\b": "himachal pradesh",
    # Indic script state transliterations
    r"मध्य\s+प्रदेश": "madhya pradesh",
    r"राजस्थान": "rajasthan",
    r"महाराष्ट्र": "maharashtra",
    r"उत्तर\s+प्रदेश": "uttar pradesh",
    r"गुजरात": "gujarat",
    r"తెలంగాణ": "telangana",
    r"ఆంధ్ర\s+ప్రదేశ్": "andhra pradesh",
    # Distinct US state abbreviations (unambiguous, skipping colliding stop words)
    r"\bcalif\b": "california",
    r"\btx\b": "texas",
    r"\bny\b": "new york",
    r"\bfl\b": "florida",
    r"\btn\b": "tennessee",
    r"\but\b": "utah",
    r"\bnc\b": "north carolina",
    r"\bmd\b": "maryland",
    r"\bva\b": "virginia",
    r"\bpa\b": "pennsylvania",
    r"\bga\b": "georgia",
    r"\bil\b": "illinois",
    r"\boh\b": "ohio",
    r"\bmi\b": "michigan",
    r"\bco\b": "colorado",
    r"\baz\b": "arizona",
    r"\bwa\b": "washington",
    r"\bma\b": "massachusetts",
    r"\bmo\b": "missouri",
    r"\bwi\b": "wisconsin",
    r"\bmn\b": "minnesota",
    r"\bsc\b": "south carolina",
    r"\bal\b": "alabama",
    r"\bla\b": "louisiana",
    r"\bky\b": "kentucky",
    r"\bor\b": "oregon",
    r"\bok\b": "oklahoma",
    r"\bct\b": "connecticut",
    r"\bia\b": "iowa",
    r"\bms\b": "mississippi",
    r"\bar\b": "arkansas",
    r"\bks\b": "kansas",
    r"\bnv\b": "nevada",
    r"\bnm\b": "new mexico",
    r"\bne\b": "nebraska",
    r"\bwv\b": "west virginia",
    r"\bid\b": "idaho",
    r"\bhi\b": "hawaii",
    r"\bnh\b": "new hampshire",
    r"\bme\b": "maine",
    r"\bri\b": "rhode island",
    r"\bmt\b": "montana",
    r"\bde\b": "delaware",
    r"\bsd\b": "south dakota",
    r"\bnd\b": "north dakota",
    r"\bak\b": "alaska",
    r"\bdc\b": "district of columbia",
    r"\bvt\b": "vermont",
    r"\bwy\b": "wyoming",
}

# Compile regular expressions once for high throughput
_COMPILED_LEGAL_SUFFIXES = [(re.compile(pat, flags=re.IGNORECASE | re.UNICODE), repl) for pat, repl in LEGAL_SUFFIXES_ORDERED]
_COMPILED_ADDRESS_ABBREV = [(re.compile(pat, flags=re.IGNORECASE | re.UNICODE), repl) for pat, repl in ADDRESS_ABBREVIATIONS.items()]


# -----------------------------------------------------------------------------
# 3. String-level Normalization Functions
# -----------------------------------------------------------------------------

def clean_base_text(text: Optional[str]) -> str:
    """Base cleaner: handles None/empty, lowercase, replaces &, strips punctuation via unicode table, collapses whitespace."""
    if text is None or not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.replace("&", " and ")
    text = text.replace("@", " at ")
    text = text.replace("+", " plus ")
    # Strip punctuation & symbols via C-level maketrans (preserves Letters, Combining Marks, Numbers)
    text = text.translate(_PUNCT_TRANSLATE)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def normalize_name(name: Optional[str]) -> str:
    """
    Normalizes a business name:
    1. Base clean (lowercase, strip punctuation, collapse whitespace).
    2. Strips URL/web domain noise if present.
    3. Normalizes legal suffixes to canonical forms (e.g., 'private limited' -> 'pvt ltd').
    """
    if name is None or not isinstance(name, str):
        return ""
    
    # Check for domain noise e.g., 'cardiologymetrocare.com' or '... | www.foo.com'
    cleaned = _URL_RE.sub(" ", name)
    cleaned = clean_base_text(cleaned)
    if not cleaned:
        cleaned = clean_base_text(name)
        
    # Standardize legal suffixes
    for pat, repl in _COMPILED_LEGAL_SUFFIXES:
        cleaned = pat.sub(repl, cleaned)
        
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned


def normalize_address(address: Optional[str]) -> str:
    """
    Normalizes a business address:
    1. Strips leading symbol noise (e.g., '##19821' -> '19821').
    2. Base clean (lowercase, strip punctuation, collapse whitespace).
    3. Expands street and state abbreviations.
    """
    if address is None or not isinstance(address, str):
        return ""
    
    # Strip leading noise
    cleaned = _PREFIX_NOISE_RE.sub("", address)
    cleaned = clean_base_text(cleaned)
    
    # Expand address abbreviations & state codes
    for pat, repl in _COMPILED_ADDRESS_ABBREV:
        cleaned = pat.sub(repl, cleaned)
        
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned


# -----------------------------------------------------------------------------
# 4. DataFrame-level Normalization
# -----------------------------------------------------------------------------

def validate_schema(df: pd.DataFrame, expected_cols: List[str] = None) -> None:
    """Validates that expected columns are present and entity_id has no nulls."""
    if expected_cols is None:
        expected_cols = ["entity_id", "business_name", "business_address", "country"]
    
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Schema validation failed: Missing columns {missing}. Found: {list(df.columns)}")
    
    if "entity_id" in df.columns and df["entity_id"].isna().any():
        null_count = df["entity_id"].isna().sum()
        raise ValueError(f"Schema validation failed: 'entity_id' column contains {null_count} null values.")


def preprocess_dataframe(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Preprocesses a business record DataFrame.
    Validates schema, normalizes names and addresses, and keeps BOTH raw and normalized columns.

    Resulting columns:
    - entity_id
    - country
    - business_name (raw)
    - business_address (raw)
    - business_name_norm (normalized)
    - business_address_norm (normalized)
    """
    validate_schema(df)
    
    out_df = df.copy()
    
    # Fill NAs in name/address with empty string
    out_df["business_name"] = out_df["business_name"].fillna("")
    out_df["business_address"] = out_df["business_address"].fillna("")
    
    # Compute normalized columns
    out_df["business_name_norm"] = out_df["business_name"].apply(normalize_name)
    out_df["business_address_norm"] = out_df["business_address"].apply(normalize_address)
    
    return out_df


def preprocess_file(input_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
    """Reads a TSV file with sep='\\t', preprocesses it, and optionally saves to output_path."""
    print(f"Loading {input_path} (sep='\\t')...")
    df = pd.read_csv(input_path, sep="\t")
    processed = preprocess_dataframe(df)
    if output_path:
        print(f"Saving preprocessed data to {output_path} (sep='\\t')...")
        processed.to_csv(output_path, sep="\t", index=False)
    return processed


if __name__ == "__main__":
    # Self-test on sample data
    test_names = [
        "Davis Family Office",
        "Davis Family Offie",
        "Clairvoyant Record Private Limited",
        "Clairvoyant Récord Private Ltd",
        "सुप्रीम आईटी प्राइवेट लिमिटेड",
        "Cardiology Metro Care Associates Inc | www.cardiology.com",
    ]
    test_addrs = [
        "88 Olive Circle, Lebanon, TN",
        "88 OLIVE CIR, LEBANON, TN",
        "1St Floor, Tamrakarmall, New Bus Stand, Sehore, Madhya Pradesh",
        "7-1ST FLOOR, TAMRAKARMALL, NEW BUS STAND, SEHORE, मध्य प्रदेश",
        "Flat No:101, Anuska Towers, Opp. Mercedes Benz Show Room, Lakdi- Ka, -Pool, Hyderabad, Telangana",
        "FLAT NO:101, -POOL, HYDERABAD, తెలంగాణ",
        "##19821 WHEELWRIGHT DR, MONTGOMERY VILLAGE, MD",
    ]
    
    print("--- Preprocessing Self-Test ---")
    print("NAMES:")
    for n in test_names:
        print(f"  RAW:  {n}")
        print(f"  NORM: {normalize_name(n)}\n")
        
    print("ADDRESSES:")
    for a in test_addrs:
        print(f"  RAW:  {a}")
        print(f"  NORM: {normalize_address(a)}\n")
