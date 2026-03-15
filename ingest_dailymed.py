import os
import glob
import time
from typing import List, Dict, Any, Optional

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

DAILYMED_DIR = "data/Dailymed data"       # folder with XML files
DAILYMED_PATTERN = "*.xml"

# For testing; set to None to process all files
MAX_FILES: Optional[int] = None

CHROMA_DIR = "chroma_healthcare"
CHROMA_COLLECTION = "healthcare_topics"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# From error message: Chroma max batch size
CHROMA_MAX_BATCH = 5461


def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


# =========================
# NEO4J SETUP
# =========================

def get_neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def init_constraints(driver):
    with driver.session() as session:
        session.execute_write(
            lambda tx: tx.run(
                "CREATE CONSTRAINT drug_id IF NOT EXISTS FOR (d:Drug) REQUIRE d.id IS UNIQUE"
            )
        )
    log("Neo4j Drug constraint ensured.")


# =========================
# CHROMA SETUP
# =========================

def get_chroma_collection():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = PersistentClient(path=CHROMA_DIR)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )

    try:
        col = client.get_collection(
            name=CHROMA_COLLECTION,
            embedding_function=embedding_fn,
        )
    except Exception:
        col = client.create_collection(
            name=CHROMA_COLLECTION,
            embedding_function=embedding_fn,
        )
    log("Chroma collection (shared with MedlinePlus) ready.")
    return col


# =========================
# XML PARSING HELPERS
# =========================

NS = {"spl": "urn:hl7-org:v3"}


def get_text(el: Optional[ET._Element]) -> str:
    if el is None:
        return ""
    return " ".join(el.itertext()).strip()


def parse_dailymed_file(path: str) -> Optional[Dict[str, Any]]:
    """
    Robust parser:
    - drug_id: setId@root
    - drug_name: first <name> under manufacturedProduct
    - label_highlights: <title> text
    """
    try:
        parser = ET.XMLParser(recover=True, huge_tree=True)
        tree = ET.parse(path, parser)
        root = tree.getroot()

        # 1) drug_id from setId@root
        setid_el = root.find(".//spl:setId", NS)
        if setid_el is None or "root" not in setid_el.attrib:
            log(f"  [SKIP] No setId root in {path}")
            return None
        drug_id = setid_el.attrib["root"].strip()

        # 2) label_highlights from <title>
        title_el = root.find(".//spl:title", NS)
        label_highlights = get_text(title_el)

        # 3) drug_name from product data section (code=48780-1) if possible
        drug_name = ""
        section_candidates = root.findall(".//spl:section", NS)
        for sec in section_candidates:
            code_el = sec.find("spl:code", NS)
            if code_el is not None and code_el.get("code") == "48780-1":
                name_el = sec.find(".//spl:manufacturedProduct//spl:name", NS)
                if name_el is not None and get_text(name_el):
                    drug_name = get_text(name_el)
                    break

        # Fallback: any manufacturedProduct name
        if not drug_name:
            name_el = root.find(".//spl:manufacturedProduct//spl:name", NS)
            if name_el is not None and get_text(name_el):
                drug_name = get_text(name_el)

        if not drug_name:
            log(f"  [SKIP] Could not find drug name in {path}")
            return None

        return {
            "id": drug_id,
            "name": drug_name,
            "highlights": label_highlights,
            "source_file": os.path.basename(path),
        }

    except Exception as e:
        log(f"  [ERROR] Parsing {path}: {e}")
        return None


# =========================
# COLLECT FILES
# =========================

def collect_dailymed_records() -> List[Dict[str, Any]]:
    pattern = os.path.join(DAILYMED_DIR, DAILYMED_PATTERN)
    files = sorted(glob.glob(pattern))
    if MAX_FILES is not None:
        files = files[:MAX_FILES]

    log(f"Found {len(files)} DailyMed XML files to process (MAX_FILES={MAX_FILES}).")

    records: List[Dict[str, Any]] = []
    for i, path in enumerate(files, start=1):
        log(f"Processing file {i}/{len(files)}: {os.path.basename(path)}")
        rec = parse_dailymed_file(path)
        if rec:
            records.append(rec)

    log(f"Parsed {len(records)} valid DailyMed records.")
    return records


# =========================
# INGEST INTO NEO4J
# =========================

def ingest_dailymed_into_neo4j(driver, records: List[Dict[str, Any]]):
    if not records:
        log("No DailyMed records to ingest into Neo4j.")
        return

    log("Ingesting DailyMed Drug nodes into Neo4j...")
    with driver.session() as session:
        session.execute_write(
            lambda tx, rows: tx.run(
                """
                UNWIND $rows AS row
                MERGE (d:Drug {id: row.id})
                SET d.name = row.name,
                    d.label_highlights = row.highlights,
                    d.source_file = row.source_file
                """,
                rows=rows
            ),
            records,
        )
    log("Neo4j Drug ingestion completed.")


# =========================
# INGEST INTO CHROMA (BATCHED)
# =========================

def ingest_dailymed_into_chroma(collection, records: List[Dict[str, Any]]):
    if not records:
        log("No DailyMed records to ingest into Chroma.")
        return

    log("Ingesting DailyMed records into Chroma (shared collection)...")
    ids = []
    docs = []
    metas = []

    for r in records:
        rid = r["id"]
        name = r["name"]
        highlights = r["highlights"] or ""
        src = r["source_file"]

        text = f"Drug: {name}\n\nHighlights: {highlights}"
        ids.append(f"drug_{rid}")  # avoid clash with Topic ids
        docs.append(text)
        metas.append({
            "type": "drug",
            "id": rid,
            "name": name,
            "source": "DailyMed",
            "file": src,
        })

    total = len(ids)
    start = 0
    batch_num = 0

    while start < total:
        end = min(start + CHROMA_MAX_BATCH, total)
        batch_num += 1
        log(f"  Upserting batch {batch_num}: records {start} to {end - 1}")
        collection.upsert(
            ids=ids[start:end],
            documents=docs[start:end],
            metadatas=metas[start:end],
        )
        start = end

    log(f"Ingested {total} DailyMed entries into Chroma in {batch_num} batches.")


# =========================
# MAIN
# =========================

def main():
    log("Starting DailyMed ingestion pipeline.")
    if not os.path.isdir(DAILYMED_DIR):
        log(f"DailyMed directory not found: {DAILYMED_DIR}")
        return

    driver = get_neo4j_driver()
    init_constraints(driver)
    chroma_collection = get_chroma_collection()

    records = collect_dailymed_records()
    if not records:
        log("No valid DailyMed records parsed; stopping.")
        return

    ingest_dailymed_into_neo4j(driver, records)
    ingest_dailymed_into_chroma(chroma_collection, records)

    log("DailyMed ingestion finished successfully.")


if __name__ == "__main__":
    main()
