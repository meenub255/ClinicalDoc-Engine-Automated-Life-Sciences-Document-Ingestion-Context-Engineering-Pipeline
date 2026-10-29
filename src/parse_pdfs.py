"""Stage 1 - Layout-aware PDF parsing with pdfplumber.

Why: naive text splitters flatten tables into word soup. Here each page is split into
(a) table regions, rendered as Markdown so row/column structure survives, and
(b) prose, extracted *outside* the table bounding boxes so table cells aren't duplicated.

Output: data/parsed/<doc>.jsonl  - one record per page element
        {doc_id, page, kind: "text"|"table", content}
"""
import json
import sys
from pathlib import Path

import pdfplumber

ROOT = Path(__file__).resolve().parent.parent
RAW, PARSED = ROOT / "data" / "raw", ROOT / "data" / "parsed"
MAX_PAGES = 120  # protocols can be 300+ pages; cap for demo speed


def table_to_markdown(rows) -> str:
    rows = [[(c or "").replace("\n", " ").strip() for c in r] for r in rows if r and any(r)]
    if len(rows) < 2 or max(len(r) for r in rows) < 2:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    md = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    md += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(md)


def parse_pdf(path: Path):
    out = []
    with pdfplumber.open(path) as pdf:
        for pno, page in enumerate(pdf.pages[:MAX_PAGES], start=1):
            tables = page.find_tables()
            bboxes = [t.bbox for t in tables]

            def outside(obj, boxes=bboxes):
                x0, top, x1, bottom = obj["x0"], obj["top"], obj["x1"], obj["bottom"]
                return not any(x0 >= b[0] - 1 and x1 <= b[2] + 1 and top >= b[1] - 1 and bottom <= b[3] + 1
                               for b in boxes)

            text = (page.filter(outside).extract_text() or "").strip() if bboxes else (page.extract_text() or "").strip()
            if text:
                out.append({"doc_id": path.stem, "page": pno, "kind": "text", "content": text})
            for t in tables:
                md = table_to_markdown(t.extract())
                if md:
                    out.append({"doc_id": path.stem, "page": pno, "kind": "table", "content": md})
    return out


def main():
    PARSED.mkdir(parents=True, exist_ok=True)
    for pdf in sorted(RAW.glob("*.pdf")):
        try:
            recs = parse_pdf(pdf)
        except Exception as e:  # noqa: BLE001
            print(f"FAILED {pdf.name}: {e}", file=sys.stderr)
            continue
        (PARSED / f"{pdf.stem}.jsonl").write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
        n_t = sum(r["kind"] == "table" for r in recs)
        print(f"{pdf.name}: {len(recs)} elements ({n_t} tables)")


if __name__ == "__main__":
    main()

