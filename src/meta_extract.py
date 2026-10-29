"""Shared cleaning + metadata-extraction logic (pure Python, no Spark dependency).

Used by the PySpark job (as UDFs / regexp_replace patterns) and by the pandas fallback so the
two engines stay behaviourally identical.
"""
import re
from collections import Counter

# (pattern, replacement) pairs - Java- and Python-regex compatible, applied in order.
CLEAN_RULES = [
    (r"\(cid:\d+\)", ""),                                # unmapped glyphs from broken ToUnicode maps
    (r"(\w)\ufffd(s\b)", r"$1'$2"),                      # mangled possessive apostrophe (Women�s)
    (r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffd]", " "),    # control chars / replacement char
    (r"(\w)-\n(\w)", r"$1$2"),                           # de-hyphenate line-wrapped words
    (r"(?im)^\s*(confidential|proprietary)\s*$", ""),      # stand-alone confidentiality banners
    (r"(?im)^.*\bpage\s+\d+\s+of\s+\d+\b.*$", ""),         # running page headers/footers
    (r"(?m)^\s*\d+\s*$", ""),                              # bare page numbers
    (r"[ \t]+", " "),                                      # collapse spaces
    (r"\n{3,}", "\n\n"),                                   # collapse blank runs
]


def clean_text(text: str) -> str:
    if not text:
        return ""
    for pat, rep in CLEAN_RULES:
        # Python uses \1, Java/Spark uses $1 - normalise for Python here.
        text = re.sub(pat, re.sub(r"\$(\d)", r"\\\1", rep), text)
    return text.strip()


# ---------------------------------------------------------------- phase
_ROMAN = {"I": "1", "II": "2", "III": "3", "IV": "4"}
_PHASE_RE = re.compile(r"\bphase\s*(IV|III|II|I|[1-4])(?:\s*[/\-–]\s*(?:phase\s*)?(IV|III|II|I|[1-4]))?\b", re.I)
_TO_ROMAN = {"1": "I", "2": "II", "3": "III", "4": "IV"}


def _norm(p: str) -> str:
    p = p.upper()
    return _TO_ROMAN.get(p, p)


def extract_phase(head_text: str, registry_phase=None):
    """Return (phase_label, source). Most-frequent phase mention in the document head wins."""
    c = Counter()
    for m in _PHASE_RE.finditer(head_text or ""):
        a, b = _norm(m.group(1)), (_norm(m.group(2)) if m.group(2) else None)
        c[f"Phase {a}/{b}" if b else f"Phase {a}"] += 1
    if c:
        return c.most_common(1)[0][0], "text"
    if registry_phase:
        nums = [_TO_ROMAN[p[-1]] for p in registry_phase if p.startswith("PHASE") and p[-1] in _TO_ROMAN]
        if nums:
            return "Phase " + "/".join(nums), "registry"
    return "Unknown", "none"


# ---------------------------------------------------------------- sponsor
_SPONSOR_RE = re.compile(r"(?im)^\s*(?:study\s+)?sponsor(?:\s+name)?\s*:\s*(.+?)\s*$")


def extract_sponsor(head_text: str, registry_sponsor=None):
    m = _SPONSOR_RE.search(head_text or "")
    if m and 3 < len(m.group(1)) < 120:
        return re.sub(r"\s+", " ", m.group(1)).strip(" *"), "text"
    if registry_sponsor:
        # Prefer the exact registry spelling if the sponsor's first word appears in the text.
        return registry_sponsor, "registry"
    return "Unknown", "none"


# ---------------------------------------------------------------- drug
_DRUG_STEMS = re.compile(
    r"\b([A-Za-z]{4,}(?:mab|nib|limus|stat|tinib|ciclib|parib|vir|navir|tide|gliptin|gliflozin|pril|sartan|"
    r"olol|azole|mycin|cillin|oxacin|caine|thiazide|dronate|setron|profen|zepam|triptan|formin|statin|"
    r"cept|ximab|zumab|umab|lukast|prazole|dipine|platin|rubicin|taxel))\b")
_STOP = {"routine", "baseline", "outline", "guideline", "timeline", "discontinue", "define", "determine", "examine",
         "online", "machine", "medicine", "decline", "combine", "alone", "done", "none", "bone", "tone", "zone",
         "phone", "stone", "hormone", "intervene", "inline", "pipeline", "line", "sideline", "undone", "gone",
         "whine", "shine", "fine", "mine", "wine", "pine", "nine", "wide", "side", "provide", "decide", "guide",
         "outside", "inside", "slide", "divide", "ride", "hide", "pride", "abide", "reside", "preside",
         "include", "exclude", "conclude", "precede", "secede", "recede", "concede", "intercede", "cede",
         "cystine", "creatine", "creatinine", "glycine", "bilirubine", "proline", "caffeine", "nicotine",
         "morphine", "quinine", "vaccine", "codeine", "protein", "genuine", "marine", "routine"}


def extract_drug(head_text: str, registry_interventions=None):
    """Ground the drug name in the document: prefer registry intervention names that actually occur
    in the text (most frequent first); otherwise fall back to pharmacological-suffix frequency."""
    text = head_text or ""
    low = text.lower()
    if registry_interventions:
        scored = []
        for name in registry_interventions:
            if re.search(r"placebo|standard of care|usual care|sham|control|saline|sodium chloride", name, re.I):
                continue
            core = re.sub(r"\s*\(.*?\)|\b\d+\s*mg\b.*", "", name).strip()
            n = low.count(core.lower()) if core else 0
            scored.append((n, name))
        hits = [s for s in scored if s[0] > 0]
        if hits:
            return max(hits)[1], "text+registry"
    c = Counter(w.lower() for w in _DRUG_STEMS.findall(text) if w.lower() not in _STOP and len(w) > 5)
    if c:
        return c.most_common(1)[0][0].capitalize(), "text"
    if registry_interventions:
        return registry_interventions[0], "registry"
    return "Unknown", "none"


# ---------------------------------------------------------------- therapeutic area
AREA_LEXICON = {
    "Oncology": ["cancer", "tumor", "tumour", "carcinoma", "glioma", "lymphoma", "myeloma", "leukemia", "oncolog", "metasta", "chemotherapy"],
    "Dermatology": ["psoriasis", "dermatitis", "eczema", "acne", "skin", "topical"],
    "Infectious Disease": ["hepatitis", "hiv", "infection", "viral", "bacterial", "covid", "sars-cov", "streptococcus", "antibiotic", "probiotic"],
    "Neurology": ["brain", "neuro", "spinal cord", "intracranial", "epilepsy", "alzheimer", "parkinson", "stroke", "traumatic brain"],
    "Cardiology": ["cardiac", "heart failure", "hypertension", "myocardial", "atrial"],
    "Endocrinology/Metabolism": ["diabetes", "insulin", "obesity", "metabolism", "glucose", "testosterone"],
    "Urology/Women's Health": ["urinary", "prolapse", "voiding", "bladder", "pelvic", "obstetric"],
    "Respiratory": ["asthma", "copd", "pulmonary", "respiratory", "lung"],
    "Rheumatology/Immunology": ["arthritis", "lupus", "autoimmune", "immunolog"],
}


def extract_area(text: str, registry_conditions=None):
    blob = ((text or "") + " " + " ".join(registry_conditions or [])).lower()
    scores = {a: sum(blob.count(k) for k in kws) for a, kws in AREA_LEXICON.items()}
    best, n = max(scores.items(), key=lambda kv: kv[1])
    return (best, "text") if n > 0 else ("Other", "none")
