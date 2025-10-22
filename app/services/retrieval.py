# app/services/retrieval.py
from typing import List, Dict, Any, Optional
from app.services.lc_vector import get_retriever

def _run_retriever(retriever, query: str):
    """
    LC >= 0.3 retrievers are Runnables -> use .invoke(query).
    Fallback to .get_relevant_documents for older versions.
    """
    try:
        return retriever.invoke(query)  # modern LC
    except AttributeError:
        return retriever.get_relevant_documents(query)  # older LC

def _build_meta_filter(city: Optional[str], specialty: Optional[str], insurer: Optional[str]) -> Dict[str, Any]:
    """
    Chroma filter:
      - exact match: {"city": {"$eq": "Jaipur"}}
      - substring contains: {"specialties": {"$contains": "cardiology"}}
    We stored specialties/insurers as pipe-strings like "cardiology|neurology".
    """
    f: Dict[str, Any] = {}
    if city and city != "All":
        f["city"] = {"$eq": city}
    if specialty and specialty != "All":
        f["specialties"] = {"$contains": specialty}
    if insurer and insurer != "All":
        f["insurers"] = {"$contains": insurer}
    return f

def search_hospitals(query: str, k: int = 5,
                     city: Optional[str] = None,
                     specialty: Optional[str] = None,
                     insurer: Optional[str] = None) -> List[Dict[str, Any]]:
    meta_filter = _build_meta_filter(city, specialty, insurer)
    retriever = get_retriever(k=k, meta_filter=meta_filter if meta_filter else None)
    docs = _run_retriever(retriever, query)
    out: List[Dict[str, Any]] = []
    for doc in docs:
        meta = doc.metadata or {}
        out.append({
            "hospital_name": meta.get("hospital_name"),
            "city": meta.get("city"),
            "address": meta.get("address"),
            "rating": meta.get("rating"),
            "phone": meta.get("phone"),
            "website": meta.get("website"),
            "snippet": (doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else "")) if doc.page_content else ""
        })
    return out
