"""Demo: metadata-filtered, access-controlled retrieval over the parsed clinical documents.

    python src/demo_query.py
"""
from rag import SecureRetriever, build_store


def show(title, docs):
    print(f"\n=== {title} ({len(docs)} hits)")
    for d in docs:
        m = d.metadata
        snippet = d.page_content.split("\n", 1)[1][:110].replace("\n", " ")
        print(f"  - {m['doc_id']} | {m['drug']} | {m['phase']} | {m['access_level']} | p.{m['page']} {m['kind']} :: {snippet}")


def main():
    store = build_store()
    print("indexed chunks:", len(store.store))

    q = "inclusion and exclusion criteria"
    show("No phase filter, public_user", SecureRetriever(store, "public_user", k=6).retrieve(q))
    show("Phase III only, public_user", SecureRetriever(store, "public_user", k=6).retrieve(q, phase="Phase III"))
    show("Phase II only, admin", SecureRetriever(store, "admin", k=6).retrieve(q, phase="Phase II"))

    q2 = "schedule of assessments table visit"
    show("Tables only, Oncology, admin", SecureRetriever(store, "admin", k=4).retrieve(q2, area="Oncology", kind="table"))

    # Authorization checks: a restricted doc (daratumumab) must be invisible to lower roles.
    for role in ("public_user", "employee", "admin"):
        hits = SecureRetriever(store, role, k=50).retrieve("daratumumab dose cyclophosphamide bortezomib")
        levels = sorted({d.metadata["access_level"] for d in hits})
        print(f"\n[{role}] 'daratumumab' query returned access levels {levels}; "
              f"leaked restricted: {'restricted' in levels and role != 'admin'}")
        assert all(d.metadata["access_level"] in {"public"} | ({"internal"} if role != "public_user" else set())
                   | ({"restricted"} if role == "admin" else set()) for d in hits)
    print("\nAll access-control assertions passed.")


if __name__ == "__main__":
    main()

