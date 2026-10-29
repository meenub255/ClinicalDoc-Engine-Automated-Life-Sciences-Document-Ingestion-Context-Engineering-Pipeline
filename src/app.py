"""Streamlit UI for the metadata-filtered clinical retriever.   streamlit run src/app.py"""
import pandas as pd
import streamlit as st
from langchain_core.documents import Document

from rag import ROLE_ACCESS, SecureRetriever, build_store

st.set_page_config(page_title="Clinical DOC Engine", layout="wide")


@st.cache_resource
def store():
    return build_store()


@st.cache_data
def docs_table():
    return pd.read_parquet("data/processed/documents.parquet")[
        ["doc_id", "drug", "phase", "therapeutic_area", "sponsor", "access_level"]]


s = store()
docs = docs_table()

st.title("Clinical DOC Engine")
st.caption("Metadata-filtered retrieval over ClinicalTrials.gov protocols")

qp = st.query_params


def idx(options, key):
    return options.index(qp[key]) if qp.get(key) in options else 0


with st.sidebar:
    st.header("Filters")
    roles = list(ROLE_ACCESS)
    role = st.selectbox("Your role (access control)", roles, index=idx(roles, "role"))
    st.caption("Allowed: " + ", ".join(sorted(ROLE_ACCESS[role])))
    phases = ["Any"] + sorted(docs.phase.unique())
    phase = st.selectbox("Trial phase", phases, index=idx(phases, "phase"))
    areas = ["Any"] + sorted(docs.therapeutic_area.unique())
    area = st.selectbox("Therapeutic area", areas, index=idx(areas, "area"))
    drug = st.text_input("Drug contains", qp.get("drug", ""))
    kinds = ["Any", "text", "table"]
    kind = st.radio("Content type", kinds, index=idx(kinds, "kind"), horizontal=True)
    k = st.slider("Results", 1, 10, 5)

query = st.text_input("Question", qp.get("q", "inclusion and exclusion criteria"))

f = dict(phase=None if phase == "Any" else phase, area=None if area == "Any" else area,
         drug=drug or None, kind=None if kind == "Any" else kind)
r = SecureRetriever(s, role, k=k)
pred = r._predicate(**f)
all_docs = [Document(page_content=v["text"], metadata=v["metadata"]) for v in s.store.values()]
eligible = sum(pred(d) for d in all_docs)
c1, c2, c3 = st.columns(3)
c1.metric("Chunks indexed", len(all_docs))
c2.metric("Chunks visible after filters", eligible)
c3.metric("Blocked by filters/access", len(all_docs) - eligible)

hits = r.retrieve(query, **f)
st.subheader(f"Top {len(hits)} results")
for d in hits:
    m = d.metadata
    with st.container(border=True):
        st.markdown(f"**{m['drug']}** · `{m['phase']}` · {m['therapeutic_area']} · `{m['access_level']}` · "
                    f"{m['doc_id']} p.{m['page']} ({m['kind']})")
        body = d.page_content.split("\n", 1)[1]
        if m["kind"] == "table":
            st.markdown(body[:700])
        else:
            st.write(body[:500])

with st.expander("Document catalogue (metadata extracted by the Spark job)"):
    st.dataframe(docs, use_container_width=True)
