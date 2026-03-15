import time

from typing import List, Tuple, Dict



from chromadb import PersistentClient

from chromadb.utils import embedding_functions

from langchain_ollama import OllamaLLM

from neo4j import GraphDatabase



# ------------- CONFIG ------------- #



# Chroma (must match your ingest scripts)

CHROMA_DIR = "chroma_healthcare"

CHROMA_COLLECTION = "healthcare_topics"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"



# Neo4j

NEO4J_URI = "bolt://localhost:7687"

NEO4J_USER = "neo4j"

NEO4J_PASSWORD = "Maithili"



# LLM via Ollama

LLM_MODEL = "llama3"





def log(msg: str):

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")





# ------------- BACKENDS ------------- #



def get_chroma_collection():

    client = PersistentClient(path=CHROMA_DIR)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(

        model_name=EMBEDDING_MODEL_NAME

    )

    collection = client.get_collection(

        name=CHROMA_COLLECTION,

        embedding_function=embedding_fn,

    )

    return collection





def get_llm():

    return OllamaLLM(model=LLM_MODEL, timeout=120)





def get_neo4j_driver():

    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))





# ------------- VECTOR RETRIEVAL ------------- #



def retrieve_vector_context(question: str, k: int = 5) -> Tuple[List[str], List[Dict]]:

    """

    Vector retrieval from Chroma. Returns (documents, metadatas).

    """

    collection = get_chroma_collection()

    results = collection.query(

        query_texts=[question],

        n_results=k,

        include=["documents", "metadatas"],

    )



    docs = results.get("documents", [[]])[0]

    metas = results.get("metadatas", [[]])[0]



    log(f"Retrieved {len(docs)} context docs from Chroma.")

    for i, m in enumerate(metas):

        if isinstance(m, dict):

            title = m.get("title") or m.get("name") or m.get("id") or "UNKNOWN"

            src = m.get("source", "UNKNOWN")

        else:

            title = "UNKNOWN"

            src = "UNKNOWN"

        log(f"  [{i}] source={src}, title={title}")



    return docs, metas





# ------------- GRAPH CONTEXT ------------- #



def build_graph_context(metas: List[Dict]) -> str:

    """

    Use metadata of the best Chroma hit to fetch structured info from Neo4j.

    Assumes ingestion stored:

      - MedlinePlus topics: metadata {id, title, language, url}

      - DailyMed drugs: metadata {type:'drug', id, name, source:'DailyMed', file}

    """

    if not metas:

        return ""



    best = metas[0]

    if not isinstance(best, dict):

        return ""



    source = best.get("source", "")

    obj_id = best.get("id")

    title = best.get("title")

    name = best.get("name")



    if not obj_id:

        return ""



    driver = get_neo4j_driver()

    graph_text = ""



    with driver.session() as session:

        if source == "DailyMed" or best.get("type") == "drug":

            # Drug context

            result = session.run(

                """

                MATCH (d:Drug {id: $id})

                RETURN d.name AS name,

                       d.label_highlights AS highlights

                """,

                id=str(obj_id),

            ).single()



            if result:

                d_name = result["name"] or name or "Unknown Drug"

                highlights = result["highlights"] or ""

                graph_text = f"Graph Drug: {d_name}\nHighlights: {highlights}"



        else:

            # Default: treat as MedlinePlus Topic

            result = session.run(

                """

                MATCH (t:Topic {id: $id})

                OPTIONAL MATCH (t)-[:IN_GROUP]->(g:TopicGroup)

                OPTIONAL MATCH (t)-[:RELATED_TO]->(rt:Topic)

                RETURN t.title AS title,

                       t.full_summary AS summary,

                       collect(DISTINCT g.name) AS groups,

                       collect(DISTINCT rt.title) AS related

                """,

                id=str(obj_id),

            ).single()



            if result:

                t_title = result["title"] or title or "Unknown Topic"

                groups = [g for g in result["groups"] if g] or []

                related = [r for r in result["related"] if r] or []



                lines = [f"Graph Topic: {t_title}"]

                if groups:

                    lines.append("Groups: " + ", ".join(groups))

                if related:

                    lines.append("Related topics: " + ", ".join(related))

                graph_text = "\n".join(lines)



    return graph_text





# ------------- ANSWER MODES ------------- #



def answer_llm_only(question: str) -> str:

    """

    Baseline: model answers from its own knowledge only.

    """

    llm = get_llm()

    prompt = f"""

You are a medical assistant. Answer this question using your general medical knowledge.

Be concise and factual.



QUESTION:

{question}



ANSWER:

"""

    return llm.invoke(prompt).strip()





def answer_vector_only(question: str) -> str:

    """

    RAG over Chroma only (MedlinePlus + DailyMed summaries).

    """

    llm = get_llm()

    docs, metas = retrieve_vector_context(question, k=5)



    if not docs:

        return "I could not find any relevant documents in the healthcare corpus."



    context_text = "\n\n--- DOCUMENT ---\n\n".join(docs)



    prompt = f"""

You are a medical assistant. Answer ONLY using the information in the healthcare summaries below.

If the answer is not clearly stated in the context, say: "I don't know based on these summaries."



QUESTION:

{question}



CONTEXT:

{context_text}



ANSWER:

"""

    return llm.invoke(prompt).strip()





def answer_graph_only(question: str) -> str:

    """

    Graph-only: use Neo4j nodes and properties, no Chroma.

    Very simple approach: substring search in titles/names.

    """

    driver = get_neo4j_driver()

    with driver.session() as session:

        # Match topics by title

        topic_result = session.run(

            """

            MATCH (t:Topic)

            WHERE toLower(t.title) CONTAINS toLower($q)

            RETURN t.title AS title, t.full_summary AS summary

            LIMIT 3

            """,

            q=question,

        )

        topics = topic_result.data()



        # Match drugs by name

        drug_result = session.run(

            """

            MATCH (d:Drug)

            WHERE toLower(d.name) CONTAINS toLower($q)

            RETURN d.name AS name, d.label_highlights AS highlights

            LIMIT 3

            """,

            q=question,

        )

        drugs = drug_result.data()



    if not topics and not drugs:

        return "I could not find any matching topics or drugs in the Neo4j graph."



    lines = []



    for t in topics:

        lines.append(f"Topic: {t['title']}")

        if t["summary"]:

            lines.append(f"Summary: {t['summary']}\n")



    for d in drugs:

        lines.append(f"Drug: {d['name']}")

        if d["highlights"]:

            lines.append(f"Highlights: {d['highlights']}\n")



    return "\n".join(lines).strip()





def answer_hybrid(question: str) -> str:

    """

    Hybrid: use vector retrieval (Chroma) + graph context (Neo4j).

    """

    llm = get_llm()

    docs, metas = retrieve_vector_context(question, k=5)



    if not docs:

        return "I could not find any relevant documents in the healthcare corpus."



    vector_context = "\n\n--- DOCUMENT ---\n\n".join(docs)

    graph_context = build_graph_context(metas)



    full_context = ""

    if graph_context:

        full_context += "STRUCTURED GRAPH FACTS:\n" + graph_context + "\n\n"

    full_context += "UNSTRUCTURED TEXT SUMMARIES:\n" + vector_context



    prompt = f"""

You are a medical assistant. Use BOTH the structured graph facts and the text summaries.

If the answer is not clearly supported by this context, say:

"I don't know based on these graph and text summaries."



QUESTION:

{question}



CONTEXT:

{full_context}



INSTRUCTIONS:

- Prefer MedlinePlus and DailyMed information when answering.

- If the graph facts add extra relevant details (groups, related topics, drugs), include them.

- Be concise and factual.



ANSWER:

"""

    return llm.invoke(prompt).strip()





# ------------- CLI MAIN ------------- #



def main():

    log("Full QA assistant (LLM-only / Vector-only / Graph-only / Hybrid)")

    while True:

        q = input("\nAsk a question (or 'exit'): ").strip()

        if not q or q.lower() in {"exit", "quit"}:

            break



        print("\n=== LLM ONLY ===")

        print(answer_llm_only(q))



        print("\n=== VECTOR ONLY (Chroma) ===")

        print(answer_vector_only(q))



        print("\n=== GRAPH ONLY (Neo4j) ===")

        print(answer_graph_only(q))



        print("\n=== HYBRID (Graph + Chroma) ===")

        print(answer_hybrid(q))





if __name__ == "__main__":

    main()