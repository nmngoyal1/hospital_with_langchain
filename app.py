# app.py
import os
import io
from pathlib import Path
from typing import Optional, Dict, Any

import streamlit as st
from functools import lru_cache
import pandas as pd

# Voice
try:
    import whisper  # openai-whisper
except Exception:
    whisper = None

from gtts import gTTS
from pydub import AudioSegment

# Optional mic component
try:
    from streamlit_mic_recorder import mic_recorder
except Exception:
    mic_recorder = None

# LangChain retrieval
from app.services.retrieval import search_hospitals


def check_ffmpeg() -> bool:
    import shutil
    return shutil.which("ffmpeg") is not None

@st.cache_resource
def load_whisper(model_name: str = "base"):
    if whisper is None:
        return None, "Whisper not installed. Install with: pip install openai-whisper"
    try:
        model = whisper.load_model(model_name)
        return model, None
    except Exception as e:
        return None, f"Failed to load Whisper model: {e}"

def stt_from_bytes(model, audio_bytes: bytes, lang_hint: Optional[str] = None) -> str:
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    audio = audio.set_channels(1).set_frame_rate(16000)
    buf = io.BytesIO()
    audio.export(buf, format="wav")
    buf.seek(0)

    tmp_path = Path("tmp_query.wav")
    with open(tmp_path, "wb") as f:
        f.write(buf.read())

    options = {}
    if lang_hint:
        options["language"] = lang_hint
    result = model.transcribe(str(tmp_path), **options)
    try:
        tmp_path.unlink(missing_ok=True)
    except Exception:
        pass
    return (result or {}).get("text", "").strip()

def tts_to_bytes(text: str, lang: str = "en") -> bytes:
    tts = gTTS(text=text, lang=lang)
    out = io.BytesIO()
    tts.write_to_fp(out)
    out.seek(0)
    return out.read()

def result_card(r: Dict[str, Any]):
    st.markdown(f"**{r.get('hospital_name','(unknown)')}** — {r.get('city','')}  "
                f"{'⭐ ' + str(r.get('rating')) if r.get('rating') not in (None, '') else ''}")
    if r.get("address"):
        st.write(r["address"])
    c1, c2 = st.columns(2)
    with c1:
        if r.get("phone"):
            st.write(f"📞 {r['phone']}")
    with c2:
        if r.get("website"):
            st.write(f"🌐 {r['website']}")
    with st.expander("Details"):
        st.write(r.get("snippet", ""))
    st.divider()

@lru_cache(maxsize=1)
def _facet_values() -> dict:
    """
    Read data/hospitals.csv (normalized) to build dropdown choices.
    """
    try:
        df = pd.read_csv("data/hospitals.csv")
    except Exception:
        return {"cities": [], "specialties": [], "insurers": []}

    def split_pipe(col):
        vals = set()
        series = df.get(col)
        if series is None:
            return []
        for s in series.fillna(""):
            for p in str(s).split("|"):
                p = p.strip()
                if p:
                    vals.add(p)
        return sorted(vals)

    cities_series = df.get("city")
    cities = sorted(set(cities_series.dropna().astype(str).str.strip())) if cities_series is not None else []

    specialties = split_pipe("specialties")
    insurers = split_pipe("insurers")
    return {"cities": cities, "specialties": specialties, "insurers": insurers}

def main():
    st.set_page_config(page_title="Hospital Voice Nearby — LangChain", page_icon="🩺", layout="wide")
    st.title("🩺 Hospital Voice Nearby — LangChain + Voice")

    ok_ffmpeg = check_ffmpeg()
    model, werr = load_whisper("base")

    with st.sidebar:
        st.header("Prerequisites")
        st.write(f"FFmpeg: {'✅' if ok_ffmpeg else '❌'}")
        if not ok_ffmpeg:
            st.caption("Install ffmpeg and add C:\\ffmpeg\\bin to PATH, then restart your terminal.")
        st.write(f"Whisper: {'✅' if (model and not werr) else '❌'}")
        if werr:
            st.caption(werr)

        st.header("Filters")
        facets = _facet_values()
        city = st.selectbox("City", options=["All"] + facets["cities"], index=0)
        specialty = st.selectbox("Specialty", options=["All"] + facets["specialties"], index=0)
        insurer = st.selectbox("Insurer", options=["All"] + facets["insurers"], index=0)

        st.header("Retriever Settings")
        k = st.slider("Top-K results", 1, 20, value=5)

        st.header("Voice Settings")
        lang_hint = st.selectbox("Language hint (optional)", ["", "en", "hi", "ar", "de"], index=0)
        tts_lang = st.selectbox("TTS language", ["en", "hi", "de", "ar"], index=0)

    tabs = st.tabs(["🔎 Text Search", "🎙️ Voice Search"])

    # Text Tab
    with tabs[0]:
        q = st.text_input("Try: 'cardiology in Jaipur' or 'insurers covering Apollo in Delhi'")
        if st.button("Search", key="text_search"):
            if not q.strip():
                st.warning("Enter a query.")
            else:
                rows = search_hospitals(q.strip(), k=k, city=city, specialty=specialty, insurer=insurer)
                if not rows:
                    st.info("No results found.")
                for r in rows:
                    result_card(r)

    # Voice Tab
    with tabs[1]:
        st.caption("Use your mic or upload an audio file (wav/mp3/ogg/webm).")
        col1, col2 = st.columns(2)
        audio_bytes = None

        with col1:
            if mic_recorder is not None:
                st.caption("Mic recorder (in-browser):")
                rec = mic_recorder(start_prompt="🎙️ Start recording", stop_prompt="⏹ Stop", key="mic1")
                if rec and "bytes" in rec:
                    audio_bytes = rec["bytes"]
                    st.audio(audio_bytes, format="audio/webm")
            else:
                st.info("Optional component 'streamlit-mic-recorder' not installed. Use file upload instead.")

        with col2:
            upload = st.file_uploader("Upload audio", type=["wav", "mp3", "ogg", "webm"], accept_multiple_files=False)
            if upload:
                audio_bytes = upload.read()
                st.audio(upload.getvalue())

        if st.button("Transcribe & Search", key="voice_search"):
            if audio_bytes is None:
                st.warning("Record or upload audio first.")
            elif not ok_ffmpeg or model is None:
                st.error("Missing FFmpeg or Whisper. See sidebar.")
            else:
                with st.spinner("Transcribing..."):
                    query_text = stt_from_bytes(model, audio_bytes, lang_hint if lang_hint else None)
                if not query_text:
                    st.warning("No speech detected or transcription failed.")
                else:
                    st.success(f"You said: {query_text!r}")
                    rows = search_hospitals(query_text, k=k, city=city, specialty=specialty, insurer=insurer)
                    if not rows:
                        st.info("No results found.")
                    for r in rows:
                        result_card(r)

                    # Optional voice summary
                    try:
                        summary = f"I found {len(rows)} results for your query."
                        audio_out = tts_to_bytes(summary, lang=tts_lang)
                        st.audio(audio_out, format="audio/mp3")
                    except Exception:
                        pass

if __name__ == "__main__":
    main()
