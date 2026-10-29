"""Stage 2 - PySpark cleaning + metadata extraction -> Parquet.

    python src/clean_spark.py                 # Spark (needs Java 8/11/17)
    python src/clean_spark.py --engine pandas # identical logic, no JVM

Reads  data/parsed/*.jsonl (+ data/raw/manifest.json registry info, data/access_overrides.json)
Writes data/processed/elements.parquet   one row per cleaned text/table element + doc metadata
       data/processed/documents.parquet  one row per document (metadata + provenance of each field)
"""
import argparse
import json
from pathlib import Path

from meta_extract import (CLEAN_RULES, clean_text, extract_area, extract_drug, extract_phase,
                          extract_sponsor)

ROOT = Path(__file__).resolve().parent.parent
PARSED, RAW, OUT = ROOT / "data" / "parsed", ROOT / "data" / "raw", ROOT / "data" / "processed"
HEAD_PAGES = 3          # phase/sponsor/drug are declared on the title/synopsis pages
AREA_CHARS = 60_000     # enough text for lexicon scoring


def load_side_inputs():
    manifest = {m["nct_id"]: m for m in json.loads((RAW / "manifest.json").read_text())}
    ov = ROOT / "data" / "access_overrides.json"
    overrides = json.loads(ov.read_text()) if ov.exists() else {}
    return manifest, overrides


def doc_metadata(doc_id, head, full, manifest, overrides):
    reg = manifest.get(doc_id, {})
    phase, ps = extract_phase(head, reg.get("registry_phase"))
    sponsor, ss = extract_sponsor(head, reg.get("registry_sponsor"))
    drug, ds = extract_drug(full[:AREA_CHARS], reg.get("registry_interventions"))
    area, As = extract_area(full[:AREA_CHARS], reg.get("registry_conditions"))
    return {
        "drug": drug, "phase": phase, "therapeutic_area": area, "sponsor": sponsor,
        "access_level": overrides.get(doc_id, "public"),
        "title": reg.get("registry_title"), "source_url": reg.get("source_url"),
        "drug_src": ds, "phase_src": ps, "sponsor_src": ss, "area_src": As,
    }


# ------------------------------------------------------------------ Spark engine
def run_spark():
    from pyspark.sql import SparkSession, Window
    from pyspark.sql import functions as F
    from pyspark.sql import types as T

    spark = (SparkSession.builder.master("local[*]").appName("clinical-doc-clean")
             .config("spark.sql.shuffle.partitions", "4").config("spark.ui.enabled", "false").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    manifest, overrides = load_side_inputs()

    df = spark.read.json(str(PARSED / "*.jsonl")).withColumn("text", F.col("content"))
    for pat, rep in CLEAN_RULES:                                # distributed cleaning
        df = df.withColumn("text", F.regexp_replace("text", pat, rep))
    df = df.withColumn("text", F.trim("text")).filter(F.length("text") > 20).drop("content")
    df = df.dropDuplicates(["doc_id", "page", "kind", "text"])

    # Per-document aggregation: head text (title/synopsis pages) + full text (ordered by page).
    ordered = df.groupBy("doc_id").agg(F.sort_array(F.collect_list(F.struct("page", "kind", "text"))).alias("els"))
    ordered = (ordered
               .withColumn("head", F.concat_ws("\n", F.expr(
                   f"transform(filter(els, e -> e.page <= {HEAD_PAGES} and e.kind = 'text'), e -> e.text)")))
               .withColumn("full", F.concat_ws("\n", F.expr("transform(els, e -> e.text)"))))

    meta_schema = T.StructType([T.StructField(k, T.StringType()) for k in
                                ["drug", "phase", "therapeutic_area", "sponsor", "access_level", "title", "source_url",
                                 "drug_src", "phase_src", "sponsor_src", "area_src"]])
    meta_udf = F.udf(lambda d, h, f: doc_metadata(d, h, f, manifest, overrides), meta_schema)
    docs = ordered.withColumn("m", meta_udf("doc_id", "head", "full")).select("doc_id", "m.*")

    stats = df.groupBy("doc_id").agg(
        F.countDistinct("page").alias("pages_with_content"),
        F.sum(F.when(F.col("kind") == "table", 1).otherwise(0)).alias("n_tables"),
        F.sum(F.length("text")).alias("n_chars"))
    documents = docs.join(stats, "doc_id")
    elements = df.join(docs, "doc_id").select(
        "doc_id", "page", "kind", "text", "drug", "phase", "therapeutic_area", "sponsor", "access_level",
        "title", "source_url")

    OUT.mkdir(parents=True, exist_ok=True)
    try:
        elements.write.mode("overwrite").parquet(str(OUT / "elements_spark"))
        documents.write.mode("overwrite").parquet(str(OUT / "documents_spark"))
    except Exception as e:  # noqa: BLE001  (Windows w/o winutils.exe)
        print("Spark native parquet write failed, using pyarrow:", str(e)[:120])
    # Single-file parquet via Arrow so downstream readers need no Spark.
    elements.toPandas().to_parquet(OUT / "elements.parquet", index=False)
    documents.toPandas().to_parquet(OUT / "documents.parquet", index=False)
    spark.stop()


# ------------------------------------------------------------------ pandas engine
def run_pandas():
    import pandas as pd
    manifest, overrides = load_side_inputs()
    rows = [json.loads(l) for f in sorted(PARSED.glob("*.jsonl"))
            for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    df["text"] = df["content"].map(clean_text)
    df = df[df["text"].str.len() > 20].drop(columns="content").drop_duplicates(["doc_id", "page", "kind", "text"])
    df = df.sort_values(["doc_id", "page"])

    meta = {}
    for doc_id, g in df.groupby("doc_id"):
        head = "\n".join(g[(g.page <= HEAD_PAGES) & (g.kind == "text")].text)
        meta[doc_id] = doc_metadata(doc_id, head, "\n".join(g.text), manifest, overrides)
    m = pd.DataFrame.from_dict(meta, orient="index").rename_axis("doc_id").reset_index()
    stats = df.groupby("doc_id").agg(pages_with_content=("page", "nunique"),
                                     n_tables=("kind", lambda s: int((s == "table").sum())),
                                     n_chars=("text", lambda s: int(s.str.len().sum()))).reset_index()
    documents = m.merge(stats, on="doc_id")
    elements = df.merge(m[["doc_id", "drug", "phase", "therapeutic_area", "sponsor", "access_level", "title",
                           "source_url"]], on="doc_id")
    OUT.mkdir(parents=True, exist_ok=True)
    elements.to_parquet(OUT / "elements.parquet", index=False)
    documents.to_parquet(OUT / "documents.parquet", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["spark", "pandas"], default="spark")
    args = ap.parse_args()
    run_spark() if args.engine == "spark" else run_pandas()
    import pandas as pd
    pd.set_option("display.width", 250, "display.max_colwidth", 30, "display.max_columns", 20)
    print(pd.read_parquet(OUT / "documents.parquet")[
        ["doc_id", "drug", "phase", "therapeutic_area", "sponsor", "access_level", "n_tables", "n_chars"]])
