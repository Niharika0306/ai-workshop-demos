import streamlit as st
import requests
import numpy as np
import pandas as pd
import time

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL = "granite-embedding:278m"


# ------------------------------------------------
# Utility Functions
# ------------------------------------------------

def get_embedding(text):

    start = time.time()

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": text
        }
    )

    response.raise_for_status()

    elapsed = round(time.time() - start, 3)

    embedding = response.json()["embedding"]

    return embedding, elapsed


def cosine_similarity(a, b):

    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


# ------------------------------------------------
# Streamlit UI
# ------------------------------------------------

st.set_page_config(
    page_title="Embeddings Workshop Demo",
    layout="wide"
)

st.title("Interactive Embeddings Demo")

st.markdown("""
This workshop demo showcases:

- Finding semantically similar text
- Cosine similarity: direction over magnitude
- Embeddings capture context
- Language independence
- Embedding dimensions

Use the examples below to explain how embeddings place meaningfully related text close together in vector space.
""")

st.markdown("---")


# ==========================================================
# SECTION 1
# ==========================================================

st.header("1. Finding Semantically Similar Text")

query = st.text_input(
    "Enter Query",
    "The customer wants to cancel their hotel reservation"
)

sentences = [
    "I need to modify my hotel booking dates.",
    "Please help me cancel my reservation for tonight.",
    "I want a refund for my train ticket.",
    "The guest would like to call off the room booking.",
    "How do I update the email on my profile?",
    "The football final is scheduled for Sunday."
]

st.write("Candidate sentences:")
for sentence in sentences:
    st.code(sentence)

if st.button("Compute Similarities", key="semantic_similarity_button"):

    query_embedding, _ = get_embedding(query)

    results = []

    for sentence in sentences:

        emb, _ = get_embedding(sentence)

        score = cosine_similarity(
            query_embedding,
            emb
        )

        results.append({
            "Sentence": sentence,
            "Cosine Similarity": round(score, 4)
        })

    df = pd.DataFrame(results)

    df = df.sort_values(
        by="Cosine Similarity",
        ascending=False
    )

    st.dataframe(df)

    st.info("""
Higher cosine similarity means the vectors point in a more similar direction.

That is why sentences about cancelling a booking can rank highly
even if they do not share the exact same words.
""")


st.markdown("---")


# ==========================================================
# SECTION 2
# ==========================================================

st.header("2. Keyword Search vs Semantic Search")

query = "I cannot log into my account"

support_docs = [
    "Password reset instructions",
    "Shipping takes 5 business days",
    "Unable to access account credentials",
    "VPN setup guide",
    "Refund policy for cancelled orders"
]


if st.button("Compare Search Types", key="search_types_button"):

    # -----------------------------
    # Keyword Search
    # -----------------------------

    keyword_results = []

    for doc in support_docs:

        common_words = (
            set(query.lower().split())
            &
            set(doc.lower().split())
        )

        keyword_results.append({
            "Document": doc,
            "Keyword Score": len(common_words)
        })

    keyword_df = pd.DataFrame(keyword_results)

    keyword_df = keyword_df.sort_values(
        by="Keyword Score",
        ascending=False
    )

    st.subheader("Keyword Search Results")

    st.dataframe(keyword_df)

    # -----------------------------
    # Semantic Search
    # -----------------------------

    query_embedding, _ = get_embedding(query)

    semantic_results = []

    for doc in support_docs:

        emb, _ = get_embedding(doc)

        score = cosine_similarity(
            query_embedding,
            emb
        )

        semantic_results.append({
            "Document": doc,
            "Semantic Similarity": round(score, 4)
        })

    semantic_df = pd.DataFrame(semantic_results)

    semantic_df = semantic_df.sort_values(
        by="Semantic Similarity",
        ascending=False
    )

    st.subheader("Semantic Search Results")

    st.dataframe(semantic_df)

    st.success("""
Semantic search understands meaning,
not just exact keywords.
""")

    st.info("""
Keyword methods such as TF-IDF are still useful when exact terms matter:
product codes, error IDs, legal clauses, API names, and rare domain-specific words.

In real systems, hybrid search often works best:
keyword search for precision + embeddings for semantic recall.
""")


st.markdown("---")


# ==========================================================
# SECTION 3
# ==========================================================

st.header("3. Embeddings Capture Context")

context_examples = [
    "The bank approved my home loan application.",
    "We sat on the bank of the river at sunset.",
    "The python script crashed because of a missing package.",
    "The python wrapped itself around the tree branch."
]

selected_text = st.selectbox(
    "Choose a sentence",
    context_examples
)

st.write("Context pair examples you can discuss:")
st.code("The bank approved my home loan application.")
st.code("We sat on the bank of the river at sunset.")
st.code("The python script crashed because of a missing package.")
st.code("The python wrapped itself around the tree branch.")

if st.button("Generate Context Embedding", key="context_embedding_button"):

    embedding, elapsed = get_embedding(selected_text)

    st.success(f"Generated in {elapsed} seconds")

    st.write("Selected Sentence:")
    st.success(selected_text)

    st.write("First 20 dimensions:")
    st.code(embedding[:20])

    st.info("""
Words like 'bank' and 'python' are ambiguous on their own.

The surrounding words tell the embedding model whether
'bank' means finance or riverside, and whether 'python'
means programming language or snake.
""")


st.markdown("---")


# ==========================================================
# SECTION 4
# ==========================================================

st.header("4. Language Independence")

english = "I need to change my flight booking to tomorrow."
german = "Ich muss meine Flugbuchung auf morgen ändern."
unrelated = "The flight has been delayed."

st.write("English sentence:")
st.code(english)

st.write("German sentence:")
st.code(german)

st.write("Unrelated sentence:")
st.code(unrelated)

if st.button("Compare Cross-Language Similarity", key="cross_language_button"):

    emb1, _ = get_embedding(english)
    emb2, _ = get_embedding(german)
    emb3, _ = get_embedding(unrelated)

    sim1 = cosine_similarity(emb1, emb2)
    sim2 = cosine_similarity(emb1, emb3)

    df = pd.DataFrame({
        "Comparison": [
            "English vs German",
            "English vs Unrelated"
        ],
        "Cosine Similarity": [
            round(sim1, 4),
            round(sim2, 4)
        ]
    })

    st.dataframe(df)

    st.info("""
Even though the words are different, the meaning is the same.

Embedding models often place translated or semantically equivalent
sentences close together in vector space.
""")


st.markdown("---")


# ==========================================================
# SECTION 5
# ==========================================================

st.header("5. Embedding Dimensions")

sample_text = (
    "The doctor prescribed rest and antibiotics for the infection."
)

st.write("Sample sentence:")
st.code(sample_text)

if st.button("Show Dimensions", key="dimensions_button_primary"):

    embedding, _ = get_embedding(sample_text)

    st.success(f"Total Dimensions: {len(embedding)}")

    st.write("First 30 values of the vector:")
    st.code(embedding[:30])

    st.info("""
An embedding is a point in a high-dimensional space.

You should not interpret one dimension as 'topic', another as 'tone',
and another as 'intent'. Meaning is distributed across the full vector.

What matters most is the relative position and direction of vectors,
which is why cosine similarity is so useful.
""")

st.markdown("---")


# ==========================================================
# SECTION 6
# ==========================================================

st.header("6. Where Embeddings Can Fail")

failure_query = (
    "The customer reports that invoice INV-2024-00987 was paid twice, "
    "and the issue appears in billing-service after job RUN-441 failed."
)

failure_docs = [
    """A general troubleshooting guide for customers who cannot sign in to the mobile banking app.
    It covers password resets, MFA failures, session timeout issues, and browser cache problems.""",
    """if invoice INV-2024-00987 appears duplicated after job RUN-441,
    verify idempotency keys, and reverse the duplicate charge.""",
    """Customer support playbook for duplicate invoice payments, failed billing jobs,
    repeated charges, refund handling, and payment reconciliation across subscription systems.""",
    """Refund operations guide for support teams handling duplicate subscription payments,
    accidental renewals, and delayed settlement notifications from payment providers.""",
    """Engineering handbook for observability dashboards, service ownership, deployment approvals,
    rollback steps, and postmortem templates across internal platform teams."""
]

st.write("Query:")
st.code(failure_query)

st.write("Candidate documents:")
for doc in failure_docs:
    st.code(doc)

if st.button("Show Failure Example", key="failure_example_button"):

    query_embedding, _ = get_embedding(failure_query)

    failure_results = []
    keyword_results = []
    hybrid_results = []

    query_terms = set(
        failure_query.lower()
        .replace(",", "")
        .replace(".", "")
        .split()
    )

    for doc in failure_docs:

        emb, _ = get_embedding(doc)
        semantic_score = cosine_similarity(query_embedding, emb)

        doc_terms = set(
            doc.lower()
            .replace(",", "")
            .replace(".", "")
            .split()
        )

        keyword_score = len(query_terms & doc_terms)
        hybrid_score = round((semantic_score * 0.7) + (keyword_score * 0.3), 4)

        failure_results.append({
            "Document": doc,
            "Semantic Similarity": round(semantic_score, 4)
        })

        keyword_results.append({
            "Document": doc,
            "Keyword Overlap": keyword_score
        })

        hybrid_results.append({
            "Document": doc,
            "Hybrid Score": hybrid_score
        })

    semantic_df = pd.DataFrame(failure_results).sort_values(
        by="Semantic Similarity",
        ascending=False
    )

    keyword_df = pd.DataFrame(keyword_results).sort_values(
        by="Keyword Overlap",
        ascending=False
    )

    hybrid_df = pd.DataFrame(hybrid_results).sort_values(
        by="Hybrid Score",
        ascending=False
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Semantic Ranking")
        st.dataframe(semantic_df)

    with col2:
        st.subheader("Keyword Ranking")
        st.dataframe(keyword_df)

    with col3:
        st.subheader("Hybrid Ranking")
        st.dataframe(hybrid_df)

    st.warning("""
Embeddings can struggle when exact identifiers matter:
invoice numbers, job IDs, error codes, service names, legal references, or product SKUs.

A semantically similar document may discuss refunds or billing in general,
but miss the exact case you care about.

Keyword search helps catch the exact identifiers, and hybrid search combines:
- semantic similarity for meaning
- keyword matching for precision
""")

st.markdown("---")


# ==========================================================
# SECTION 7
# ==========================================================

st.header("7. Large Documents Need Chunking")

long_doc = """
Acme Cloud Platform Handbook

The first part of this handbook explains company history, office policies,
travel reimbursement, cafeteria rules, holiday calendars, laptop requests,
security badges, and onboarding checklists for new employees.

The second part describes engineering workflows, sprint planning,
incident response, deployment approvals, rollback procedures,
observability dashboards, and service ownership responsibilities.

The final section explains customer billing incidents. If a payment fails
with error code E401, the recommended action is to verify whether the token
used by payment-service has expired, refresh credentials, and retry the request.
"""

chunk_query = "What should I do when payment-service returns error code E401?"

st.write("Long document:")
st.code(long_doc)

st.write("Question:")
st.code(chunk_query)

if st.button("Demonstrate Chunking", key="chunking_button"):

    full_doc_embedding, _ = get_embedding(long_doc)
    query_embedding, _ = get_embedding(chunk_query)

    full_doc_score = cosine_similarity(query_embedding, full_doc_embedding)

    chunks = [
        "Company history, office policies, travel reimbursement, cafeteria rules, holiday calendars, laptop requests, security badges, and onboarding checklists.",
        "Engineering workflows, sprint planning, incident response, deployment approvals, rollback procedures, observability dashboards, and service ownership responsibilities.",
        "Customer billing incidents: if payment-service returns error code E401, verify whether the token has expired, refresh credentials, and retry the request."
    ]

    chunk_results = []
    for idx, chunk in enumerate(chunks, start=1):
        chunk_embedding, _ = get_embedding(chunk)
        score = cosine_similarity(query_embedding, chunk_embedding)
        chunk_results.append({
            "Chunk": f"Chunk {idx}",
            "Cosine Similarity": round(score, 4),
            "Text": chunk
        })

    chunk_df = pd.DataFrame(chunk_results).sort_values(
        by="Cosine Similarity",
        ascending=False
    )

    st.subheader("Full Document Score")
    st.dataframe(pd.DataFrame([{
        "Document": "Entire handbook as one embedding",
        "Cosine Similarity": round(full_doc_score, 4)
    }]))

    st.subheader("Chunked Scores")
    st.dataframe(chunk_df)

    st.info("""
When documents are too large, one embedding may blur together many topics.

Chunking helps because the query can match the specific part of the document
that actually contains the answer.
""")

st.markdown("---")




st.markdown("---")

st.success("""
Embeddings are the bridge between
human language and machine understanding.
""")