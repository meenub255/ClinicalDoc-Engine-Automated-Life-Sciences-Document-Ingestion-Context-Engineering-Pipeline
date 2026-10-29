"""Download public clinical-trial protocol / SAP PDFs from ClinicalTrials.gov (API v2).

Usage: python src/download_docs.py --n 8
Writes PDFs to data/raw/ and a manifest (data/raw/manifest.json) with registry metadata
(sponsor, phase, conditions, interventions) that is later used to validate extraction.
"""
import argparse
import json
from pathlib import Path

import requests

API = "https://clinicaltrials.gov/api/v2/studies"
HEADERS = {"User-Agent": "Mozilla/5.0 (clinical-doc-engine demo)"}
RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

# Spread across phases so phase-based filtering is meaningful.
PHASE_QUERIES = ["PHASE1", "PHASE2", "PHASE3", "PHASE3", "PHASE2", "PHASE3", "PHASE4", "PHASE2"]


def find_candidates(phase: str, page_size: int = 15):
    params = {
        "query.term": f"AREA[LargeDocHasProtocol]true AND AREA[Phase]{phase}",
        "filter.overallStatus": "COMPLETED",
        "pageSize": page_size,
    }
    r = requests.get(API, params=params, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.json().get("studies", [])


def main(n: int):
    RAW.mkdir(parents=True, exist_ok=True)
    manifest, seen = [], set()
    for phase in PHASE_QUERIES:
        if len(manifest) >= n:
            break
        for s in find_candidates(phase):
            ps = s["protocolSection"]
            nct = ps["identificationModule"]["nctId"]
            if nct in seen:
                continue
            docs = s.get("documentSection", {}).get("largeDocumentModule", {}).get("largeDocs", [])
            prot = next((d for d in docs if d.get("hasProtocol") and d["filename"].lower().endswith(".pdf")), None)
            if not prot:
                continue
            url = f"https://clinicaltrials.gov/ProvidedDocs/{nct[-2:]}/{nct}/{prot['filename']}"
            try:
                resp = requests.get(url, headers=HEADERS, timeout=120)
                resp.raise_for_status()
                if not resp.content.startswith(b"%PDF") or len(resp.content) > 15_000_000:
                    continue
            except Exception as e:  # noqa: BLE001
                print(f"skip {nct}: {e}")
                continue
            path = RAW / f"{nct}.pdf"
            path.write_bytes(resp.content)
            seen.add(nct)
            manifest.append({
                "nct_id": nct,
                "file": path.name,
                "source_url": url,
                "registry_title": ps["identificationModule"].get("briefTitle"),
                "registry_sponsor": ps["sponsorCollaboratorsModule"]["leadSponsor"]["name"],
                "registry_phase": ps["designModule"].get("phases"),
                "registry_conditions": ps.get("conditionsModule", {}).get("conditions", []),
                "registry_interventions": [
                    i["name"] for i in ps.get("armsInterventionsModule", {}).get("interventions", [])
                    if i.get("type") in ("DRUG", "BIOLOGICAL", "COMBINATION_PRODUCT", "DIETARY_SUPPLEMENT")
                ] or [i["name"] for i in ps.get("armsInterventionsModule", {}).get("interventions", [])],
            })
            print(f"downloaded {nct} ({len(resp.content)//1024} KB) phase={phase}")
            break  # one doc per phase query -> diversity
    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"{len(manifest)} documents; manifest written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    main(ap.parse_args().n)
