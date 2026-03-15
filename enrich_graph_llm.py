import time
import json
from typing import List, Dict, Any, Optional

from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from neo4j import GraphDatabase

# -------- CONFIG -------- #

CHROMA_DIR = "chroma_healthcare"
CHROMA_COLLECTION = "healthcare_topics"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Maithili"

LLM_MODEL = "llama3"
LLM_TIMEOUT = 120

# Start with a small subset to test; set to None later for full run
MAX_DOCS: Optional[int] = None # None => all docs in collection


def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


# -------- BACKENDS -------- #

def get_chroma_collection():
    client = PersistentClient(path=CHROMA_DIR)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )
    col = client.get_collection(
        name=CHROMA_COLLECTION,
        embedding_function=embedding_fn,
    )
    return col


def get_llm():
    return OllamaLLM(model=LLM_MODEL, timeout=LLM_TIMEOUT)


def get_neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# -------- LLM PROMPT & PARSER -------- #

def get_extraction_prompt():
    return PromptTemplate.from_template(
        """
You are an expert in extracting medical information. From the text below, extract entities and relationships.

Entity types: "Disease", "Symptom", "Treatment", "Prevention", "Drug", "Condition", "Intervention", "Study".
Relationship types: "HAS_SYMPTOM", "TREATED_BY", "PREVENTED_BY", "HAS_SIDE_EFFECT", "INTERACTS_WITH", "TESTS_DRUG_FOR_DISEASE".

Text:
{text_chunk}

Respond ONLY with a JSON object with two keys: "entities" and "relationships".
Example:
{{"entities": [{{"id": "Common Cold", "type": "Disease"}}],
  "relationships": [{{"source": "Common Cold", "target": "Sore Throat", "type": "HAS_SYMPTOM"}}]}}
"""
    )


def extract_knowledge(llm, prompt, text_chunk: str) -> Optional[Dict[str, Any]]:
    if not text_chunk or not text_chunk.strip():
        log("  - Skipped empty text chunk.")
        return None

    prompt_with_text = prompt.format(text_chunk=text_chunk)
    resp = llm.invoke(prompt_with_text)

    try:
        start = resp.find("{")
        end = resp.rfind("}")
        if start == -1 or end == -1:
            log("  - No JSON object found in LLM response.")
            return None
        data = json.loads(resp[start : end + 1])
        return data
    except Exception as e:
        log(f"  - Error parsing LLM response: {e}")
        return None


# -------- NEO4J POPULATION -------- #

def init_llm_constraints(driver):
    """
    Ensure uniqueness per entity label for LLM-derived nodes.
    Safe to re-run.
    """
    cyphers = [
        "CREATE CONSTRAINT disease_id IF NOT EXISTS FOR (d:Disease) REQUIRE d.id IS UNIQUE",
        "CREATE CONSTRAINT symptom_id IF NOT EXISTS FOR (s:Symptom) REQUIRE s.id IS UNIQUE",
        "CREATE CONSTRAINT treatment_id IF NOT EXISTS FOR (t:Treatment) REQUIRE t.id IS UNIQUE",
        "CREATE CONSTRAINT prevention_id IF NOT EXISTS FOR (p:Prevention) REQUIRE p.id IS UNIQUE",
        "CREATE CONSTRAINT drug_llm_id IF NOT EXISTS FOR (d:DrugLLM) REQUIRE d.id IS UNIQUE",
        "CREATE CONSTRAINT condition_id IF NOT EXISTS FOR (c:Condition) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT intervention_id IF NOT EXISTS FOR (i:Intervention) REQUIRE i.id IS UNIQUE",
        "CREATE CONSTRAINT study_id IF NOT EXISTS FOR (s:Study) REQUIRE s.id IS UNIQUE",
    ]
    with driver.session() as session:
        for c in cyphers:
            session.execute_write(lambda tx, q: tx.run(q), c)
    log("Neo4j LLM entity constraints ensured.")


def map_entity_label(e_type: str) -> Optional[str]:
    """
    Map LLM 'type' to Neo4j label.
    Use DrugLLM so we don't clash with XML Drug nodes.
    """
    t = (e_type or "").strip()
    mapping = {
        "Disease": "Disease",
        "Symptom": "Symptom",
        "Treatment": "Treatment",
        "Prevention": "Prevention",
        "Drug": "DrugLLM",
        "Condition": "Condition",
        "Intervention": "Intervention",
        "Study": "Study",
    }
    return mapping.get(t)


def populate_neo4j_from_extraction(driver, extracted: Dict[str, Any]):
    entities = extracted.get("entities", []) or []
    relationships = extracted.get("relationships", []) or []

    with driver.session() as session:
        # 1) Entities
        for e in entities:
            if not isinstance(e, dict):
                continue
            eid = str(e.get("id", "")).strip()
            etype = str(e.get("type", "")).strip()
            label = map_entity_label(etype)
            if not eid or not label:
                continue

            cypher = f"MERGE (e:{label} {{id: $id}})"
            session.execute_write(lambda tx, q, p: tx.run(q, **p), cypher, {"id": eid})

        # 2) Relationships
        for r in relationships:
            if not isinstance(r, dict):
                continue
            src = str(r.get("source", "")).strip()
            tgt = str(r.get("target", "")).strip()
            rtype = str(r.get("type", "")).strip()
            if not src or not tgt or not rtype:
                continue

            cypher = f"""
            MATCH (a {{id: $src}}), (b {{id: $tgt}})
            MERGE (a)-[r:{rtype}]->(b)
            """
            session.execute_write(
                lambda tx, q, p: tx.run(q, **p),
                cypher,
                {"src": src, "tgt": tgt},
            )


# -------- MAIN PIPELINE -------- #

def main():
    log("Starting LLM-based graph enrichment from Chroma.")

    chroma_collection = get_chroma_collection()
    llm = get_llm()
    driver = get_neo4j_driver()
    init_llm_constraints(driver)
    extraction_prompt = get_extraction_prompt()

    # Get all docs from Chroma
    data = chroma_collection.get()
    all_ids: List[str] = data["ids"]
    all_docs: List[str] = data["documents"]

    total_docs = len(all_ids)
    log(f"Total documents in Chroma: {total_docs}")

    if MAX_DOCS is not None:
        total_to_process = min(MAX_DOCS, total_docs)
    else:
        total_to_process = total_docs

    log(f"Will process {total_to_process} documents (MAX_DOCS={MAX_DOCS}).")

    for idx in range(total_to_process):
        doc_id = all_ids[idx]
        text = all_docs[idx]
        log(f"Processing doc {idx+1}/{total_to_process}: id={doc_id}")

        extracted = extract_knowledge(llm, extraction_prompt, text)
        if not extracted:
            continue

        populate_neo4j_from_extraction(driver, extracted)

    log("LLM-based graph enrichment finished.")


if __name__ == "__main__":
    main()
