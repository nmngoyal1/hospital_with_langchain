import csv
import sys
from typing import Any, Dict, List
from pathlib import Path

# Ensure project root on path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from langchain.docstore.document import Document
from app.services.lc_vector import build_vectorstore_from_documents, normalize_metadata


def safe_float(x, default=0.0):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default

def to_list(s: str) -> List[str]:
    if not s:
        return []
    return [itm.strip() for itm in s.split("|") if itm.strip()]

def row_to_doc(row: Dict[str, str]) -> Document:
    hosp = (row.get("hospital_name") or "").strip()
    city = (row.get("city") or "").strip()
    specialties = to_list(row.get("specialties", ""))
    insurers = to_list(row.get("insurers", ""))

    text = (
        f"{hosp} in {city} offers services in "
        f"{', '.join(specialties) if specialties else 'general care'} "
        f"and accepts {', '.join(insurers) if insurers else 'various insurers'}. "
        f"Address: {row.get('address','').strip()}."
    )

    meta = {
        "hospital_name": hosp,
        "address": (row.get("address") or "").strip(),
        "city": city,
        "lat": safe_float(row.get("latitude")),
        "lon": safe_float(row.get("longitude")),
        "specialties": specialties,  # list → normalized later
        "insurers": insurers,        # list → normalized later
        "rating": safe_float(row.get("rating")),
        "phone": (row.get("phone") or "").strip(),
        "website": (row.get("website") or "").strip(),
    }
    meta = normalize_metadata(meta)

    # Add tags inline to help recall
    if specialties:
        text += f"\nTAGS: {', '.join(specialties)}"

    return Document(page_content=text, metadata=meta)

def main():
    csv_path = Path("data/hospitals.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at {csv_path.resolve()}")

    docs: List[Document] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for _i, row in enumerate(reader, start=1):
            docs.append(row_to_doc(row))

    build_vectorstore_from_documents(docs)
    print(f"Ingested {len(docs)} hospitals into LangChain+Chroma.")

if __name__ == "__main__":
    main()
