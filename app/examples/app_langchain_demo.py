import streamlit as st
from app.services.retrieval import search_hospitals

st.title("Hospital Voice Nearby — LangChain Demo")
q = st.text_input("Search hospitals (e.g., 'cardiology in Jaipur')")
k = st.slider("Top-K", 1, 20, 5)
if st.button("Search"):
    rows = search_hospitals(q or "", k=k)
    for r in rows:
        st.markdown(f"**{r['hospital_name']}** — {r['city']}  ⭐ {r.get('rating','')}")
        st.write(r.get('address',''))
        if r.get('phone'): st.write(f"📞 {r['phone']}")
        if r.get('website'): st.write(f"🌐 {r['website']}")
        with st.expander("Details"):
            st.write(r['snippet'])
        st.divider()
