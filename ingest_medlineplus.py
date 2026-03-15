import os
import time
from typing import List, Dict, Any

from lxml import etree as ET
from neo4j import GraphDatabase
from chromadb import PersistentClient
from chromadb.utils import embedding_functions

# =========================
# CONFIG
# =========================

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Maithili"

MEDLINEPLUS_XML = "data/medilineplus/health_topics.xml"  

CHROMA_DIR = "chroma_healthcare"
CHROMA_COLLECTION = "healthcare_topics"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


# =========================
# NEO4J SETUP
# =========================

def get_neo4j_driver():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return driver

def init_constraints(driver):
    cyphers = [
        "CREATE CONSTRAINT topic_id IF NOT EXISTS FOR (t:Topic) REQUIRE t.id IS UNIQUE",
        "CREATE CONSTRAINT group_id IF NOT EXISTS FOR (g:TopicGroup) REQUIRE g.id IS UNIQUE",
        "CREATE CONSTRAINT mesh_id IF NOT EXISTS FOR (m:MeshDescriptor) REQUIRE m.id IS UNIQUE"
    ]
    with driver.session() as session:
        for c in cyphers:
            session.execute_write(lambda tx, q: tx.run(q), c)
    log("Neo4j constraints ensured.")


# =========================
# CHROMA SETUP
# =========================

def get_chroma_collection():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = PersistentClient(path=CHROMA_DIR)

    # FIX: use only model_name (no explicit model argument)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )

    try:
        collection = client.get_collection(
            name=CHROMA_COLLECTION,
            embedding_function=embedding_fn
        )
    except Exception:
        collection = client.create_collection(
            name=CHROMA_COLLECTION,
            embedding_function=embedding_fn
        )
    log("Chroma collection ready.")
    return collection


# =========================
# XML PARSING
# =========================

def parse_medlineplus_xml(xml_path: str):
    """
    Parse MedlinePlus health-topics XML and return list of topic dicts.
    Structure based on official description. [web:29]
    """
    log(f"Parsing XML: {xml_path}")
    parser = ET.XMLParser(recover=True)
    tree = ET.parse(xml_path, parser)
    root = tree.getroot()

    topics: List[Dict[str, Any]] = []
    for topic_el in root.findall("health-topic"):
        topic_id = topic_el.get("id")
        title = (topic_el.get("title") or "").strip()
        url = (topic_el.get("url") or "").strip()
        language = (topic_el.get("language") or "").strip()
        date_created = (topic_el.get("date-created") or "").strip()
        meta_desc = (topic_el.get("meta-desc") or "").strip()

        # also-called synonyms
        also_called = [
            ac.text.strip()
            for ac in topic_el.findall("also-called")
            if ac is not None and ac.text
        ]

        # full-summary content (escaped HTML)
        full_summary_el = topic_el.find("full-summary")
        full_summary = ""
        if full_summary_el is not None:
            if full_summary_el.text:
                full_summary = full_summary_el.text.strip()
            else:
                # fallback: concatenate all text
                full_summary = "".join(full_summary_el.itertext()).strip()

        # groups (categories) [web:29]
        groups = []
        for g in topic_el.findall("group"):
            gid = g.get("id")
            gurl = (g.get("url") or "").strip()
            gname = (g.text or "").strip()
            if gid and gname:
                groups.append({"id": gid, "name": gname, "url": gurl})

        # MeSH descriptors (controlled vocabulary) [web:29]
        mesh_descriptors = []
        for mh in topic_el.findall("mesh-heading"):
            d = mh.find("descriptor")
            if d is not None:
                mid = d.get("id")
                mname = (d.text or "").strip()
                if mid and mname:
                    mesh_descriptors.append({"id": mid, "name": mname})

        # related-topic (links between topics) [web:29]
        related_topics = []
        for rt in topic_el.findall("related-topic"):
            rid = rt.get("id")
            rurl = (rt.get("url") or "").strip()
            rtitle = (rt.text or "").strip()
            if rid and rtitle:
                related_topics.append({"id": rid, "title": rtitle, "url": rurl})

        topics.append({
            "id": topic_id,
            "title": title,
            "url": url,
            "language": language,
            "date_created": date_created,
            "meta_desc": meta_desc,
            "also_called": also_called,
            "full_summary": full_summary,
            "groups": groups,
            "mesh_descriptors": mesh_descriptors,
            "related_topics": related_topics
        })

    log(f"Parsed {len(topics)} topics from XML.")
    return topics


# =========================
# INGEST INTO NEO4J
# =========================

def ingest_topics_into_neo4j(driver, topics: List[Dict[str, Any]]):
    """
    Upsert topics, groups, MeSH descriptors and relationships.
    Re-running this is safe and will not delete data. [web:47][web:73]
    """
    log("Ingesting topics into Neo4j...")
    with driver.session() as session:

        # 1) Topic nodes
        session.execute_write(
            lambda tx, rows: tx.run(
                """
                UNWIND $rows AS row
                MERGE (t:Topic {id: row.id})
                SET t.title = row.title,
                    t.url = row.url,
                    t.language = row.language,
                    t.date_created = row.date_created,
                    t.meta_desc = row.meta_desc,
                    t.full_summary = row.full_summary
                """,
                rows=rows
            ),
            topics
        )
        log("Topics upserted.")

        # 2) TopicGroup nodes + IN_GROUP relationships
        group_rows = []
        for t in topics:
            for g in t["groups"]:
                group_rows.append({
                    "topic_id": t["id"],
                    "group_id": g["id"],
                    "group_name": g["name"],
                    "group_url": g["url"]
                })

        if group_rows:
            session.execute_write(
                lambda tx, rows: tx.run(
                    """
                    UNWIND $rows AS row
                    MERGE (g:TopicGroup {id: row.group_id})
                    SET g.name = row.group_name,
                        g.url = row.group_url
                    WITH row, g
                    MATCH (t:Topic {id: row.topic_id})
                    MERGE (t)-[:IN_GROUP]->(g)
                    """,
                    rows=rows
                ),
                group_rows
            )
            log(f"Groups upserted and linked for {len(group_rows)} rows.")

        # 3) MeshDescriptor nodes + HAS_MESH relationships
        mesh_rows = []
        for t in topics:
            for m in t["mesh_descriptors"]:
                mesh_rows.append({
                    "topic_id": t["id"],
                    "mesh_id": m["id"],
                    "mesh_name": m["name"]
                })

        if mesh_rows:
            session.execute_write(
                lambda tx, rows: tx.run(
                    """
                    UNWIND $rows AS row
                    MERGE (m:MeshDescriptor {id: row.mesh_id})
                    SET m.name = row.mesh_name
                    WITH row, m
                    MATCH (t:Topic {id: row.topic_id})
                    MERGE (t)-[:HAS_MESH]->(m)
                    """,
                    rows=rows
                ),
                mesh_rows
            )
            log(f"MeSH descriptors upserted and linked for {len(mesh_rows)} rows.")

        # 4) RELATED_TO relationships between topics
        related_rows = []
        for t in topics:
            for rt in t["related_topics"]:
                related_rows.append({
                    "topic_id": t["id"],
                    "related_id": rt["id"],
                    "related_title": rt["title"],
                    "related_url": rt["url"]
                })

        if related_rows:
            session.execute_write(
                lambda tx, rows: tx.run(
                    """
                    UNWIND $rows AS row
                    MERGE (rt:Topic {id: row.related_id})
                    SET rt.title = coalesce(rt.title, row.related_title),
                        rt.url   = coalesce(rt.url,   row.related_url)
                    WITH row, rt
                    MATCH (t:Topic {id: row.topic_id})
                    MERGE (t)-[:RELATED_TO]->(rt)
                    """,
                    rows=rows
                ),
                related_rows
            )
            log(f"Related topic relationships created for {len(related_rows)} rows.")

    log("Neo4j ingestion completed.")


# =========================
# INGEST INTO CHROMA
# =========================

def ingest_topics_into_chroma(collection, topics: List[Dict[str, Any]]):
    """
    Store topic summaries in Chroma using topic id as document id.
    Re-running will upsert (update) existing ids. [web:60][web:74]
    """
    log("Ingesting topics into Chroma...")
    ids = []
    texts = []
    metadatas = []

    for t in topics:
        tid = t["id"]
        title = t["title"]
        full_summary = t["full_summary"] or ""
        meta_desc = t["meta_desc"] or ""
        language = t["language"]
        url = t["url"]

        if not full_summary and not meta_desc:
            continue

        doc_text = f"Title: {title}\n\nSummary: {full_summary}\n\nMeta: {meta_desc}"
        ids.append(str(tid))
        texts.append(doc_text)
        metadatas.append({
            "id": tid,
            "title": title,
            "language": language,
            "url": url
        })

    if ids:
        collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
        log(f"Ingested {len(ids)} topics into Chroma.")
    else:
        log("No non-empty topics to ingest into Chroma.")


# =========================
# MAIN
# =========================

def main():
    log("Starting MedlinePlus ingestion pipeline (topics only).")
    if not os.path.exists(MEDLINEPLUS_XML):
        log(f"XML file not found: {MEDLINEPLUS_XML}")
        return

    driver = get_neo4j_driver()
    init_constraints(driver)
    chroma_collection = get_chroma_collection()

    topics = parse_medlineplus_xml(MEDLINEPLUS_XML)
    if not topics:
        log("No topics parsed; nothing to ingest.")
        return

    ingest_topics_into_neo4j(driver, topics)
    ingest_topics_into_chroma(chroma_collection, topics)

    log("Ingestion finished successfully.")

if __name__ == "__main__":
    main()
