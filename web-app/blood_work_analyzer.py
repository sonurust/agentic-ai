import os
import json
import re
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# ----------------------------
# Page configuration
# ----------------------------
st.set_page_config(
    page_title="Blood Work Analyzer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# Styling
# ----------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.6rem; padding-bottom: 2.5rem; }
    .app-subtitle { color: #9ca3af; margin-top: -0.6rem; }
    .report-box {
        max-height: 360px;
        overflow-y: auto;
        padding: 14px 16px;
        border: 1px solid rgba(128,128,128,.35);
        border-radius: 10px;
        background: rgba(128,128,128,.07);
        font-size: .92rem;
        line-height: 1.6;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
    }
    .small-muted { color: #9ca3af; font-size: .82rem; }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 10px;
        padding: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Prompts
# ----------------------------
SYSTEM_PROMPT = """
You are a careful medical laboratory report extraction and explanation assistant.
Extract every test and result that is visible in the supplied report. Preserve the
original test names, values, units, reference ranges, dates, and report wording.
Do not invent missing information. Mark a result as high or low only when the
report provides a reference range or the comparison is otherwise explicitly supported.
Explain findings in plain language and distinguish measured facts from possible
clinical interpretations. Do not provide a definitive diagnosis or prescribe treatment.
Mention that laboratory results must be interpreted by a qualified clinician in context.
Return valid JSON only, with no Markdown fences and no text before or after the JSON.
"""

OUTPUT_SCHEMA = {
    "patient_information": {
        "name": None,
        "age": None,
        "gender": None,
        "report_date": None,
        "laboratory": None,
    },
    "test_results": [
        {
            "section": "",
            "test_name": "",
            "value": "",
            "unit": "",
            "reference_range": "",
            "status": "normal | high | low | not_stated",
            "source_note": "",
        }
    ],
    "abnormal_findings": [
        {
            "test_name": "",
            "reported_value": "",
            "why_flagged": "",
        }
    ],
    "parameter_explanations": [
        {
            "test_name": "",
            "plain_language_explanation": "",
            "possible_significance": "",
        }
    ],
    "report_comments_and_recommendations": [],
    "follow_up_tests_mentioned_in_report": [],
    "limitations": [],
    "medical_safety_note": "",
}

def extract_pdf_text(uploaded_file) -> str:
    """Extract selectable text from a PDF. Scanned PDFs may need OCR."""
    reader = PdfReader(uploaded_file)
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(f"--- PAGE {page_number} ---\n{text.strip()}")
    return "\n\n".join(pages).strip()


def extract_json_from_response(content: Any) -> dict:
    """Parse JSON even if the model accidentally adds code fences."""
    if isinstance(content, list):
        content = "\n".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    text = str(content).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise ValueError("The model did not return valid JSON. Try again.")


def analyze_report(report_text: str, api_key: str, model_name: str) -> dict:
    llm = ChatNVIDIA(
        base_url="https://integrate.api.nvidia.com/v1",
        model=model_name,
        api_key=api_key,
        temperature=0.2,
        top_p=0.7,
        max_completion_tokens=16384,
        seed=42,
        timeout=240,
    )

    human_prompt = f"""
Analyze the medical report text below.

Return a JSON object matching this structure. Include every test result found.
Use an empty array when a section has no data. Use null for unavailable patient details.
Do not create example values or fill in tests that are not present.

EXPECTED JSON STRUCTURE:
{json.dumps(OUTPUT_SCHEMA, ensure_ascii=False, indent=2)}

MEDICAL REPORT TEXT:
<report>
{report_text}
</report>
"""
    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=human_prompt),
        ]
    )
    return extract_json_from_response(response.content)


# ----------------------------
# Header and sidebar
# ----------------------------
st.title("Blood Work Analyzer")
st.markdown(
    '<p class="app-subtitle">Upload a laboratory report to extract test results, '
    'identify report-flagged abnormalities, and understand the findings.</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Configuration")
    model_name = st.text_input(
        "NVIDIA model",
        value="nvidia/nemotron-3.5-super-vl-120b-a12b",
        help="Use a model available to your NVIDIA API account.",
    )
    api_key_input = st.text_input(
        "NVIDIA API key",
        value=os.getenv("NVIDIA_API_KEY", ""),
        type="password",
        help="You can also set NVIDIA_API_KEY in your .env file.",
    )
    st.divider()
    st.caption("Supported input: PDF with selectable text.")
    st.caption("Scanned/image-only PDFs require OCR before analysis.")

# ----------------------------
# Main layout
# ----------------------------
left, right = st.columns([1.05, 1], gap="large")

with left:
    st.subheader("1. Upload report")
    uploaded_pdf = st.file_uploader(
        "Choose a blood or laboratory report (PDF)",
        type=["pdf"],
        help="The report is processed for text extraction and sent to the configured model.",
    )

    if uploaded_pdf:
        st.success(f"Uploaded: {uploaded_pdf.name}")
        if st.button("Extract report text", use_container_width=True):
            try:
                extracted_text = extract_pdf_text(uploaded_pdf)
                st.session_state["blood_report_text"] = extracted_text
                st.session_state["blood_report_name"] = uploaded_pdf.name
                st.session_state.pop("analysis_result", None)
                if not extracted_text:
                    st.warning(
                        "No selectable text was found. This may be a scanned PDF; "
                        "run OCR on it before analysis."
                    )
                else:
                    st.success("Text extraction completed.")
            except Exception as exc:
                st.error(f"Could not read this PDF: {exc}")

    report_text = st.text_area(
        "Extracted report text (editable)",
        value=st.session_state.get("blood_report_text", ""),
        height=300,
        placeholder="Upload a PDF and select 'Extract report text', or paste report text here.",
    )
    st.session_state["blood_report_text"] = report_text

    analyze_clicked = st.button(
        "Analyze blood report",
        type="primary",
        use_container_width=True,
        disabled=not report_text.strip() or not api_key_input.strip(),
    )

with right:
    st.subheader("2. Analysis")
    st.info(
        "The app extracts the report text and asks the model to return structured JSON. "
        "Verify every extracted value against the original PDF."
    )
    if st.session_state.get("analysis_result"):
        st.success("An analysis is available. You can run the analysis again after editing the text.")

if analyze_clicked:
    if not api_key_input.strip():
        st.error("Enter your NVIDIA API key in the sidebar or add it to your .env file.")
    else:
        with st.spinner("Analyzing report..."):
            try:
                result = analyze_report(report_text, api_key_input.strip(), model_name.strip())
                st.session_state["analysis_result"] = result
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")

# ----------------------------
# Results
# ----------------------------
result = st.session_state.get("analysis_result")
if result:
    st.divider()
    st.header("3. Report analysis")

    patient = result.get("patient_information") or {}
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Patient", patient.get("name") or "Not stated")
    p2.metric("Age", patient.get("age") or "Not stated")
    p3.metric("Gender", patient.get("gender") or "Not stated")
    p4.metric("Report date", patient.get("report_date") or "Not stated")

    tabs = st.tabs(
        [
            "All test results",
            "Abnormal findings",
            "Explanations",
            "Comments & follow-up",
            "Raw JSON",
        ]
    )

    with tabs[0]:
        st.subheader("Extracted test results")
        test_results = result.get("test_results") or []
        if test_results:
            st.dataframe(test_results, use_container_width=True, hide_index=True)
        else:
            st.warning("No test results were returned.")

    with tabs[1]:
        st.subheader("Abnormal findings")
        abnormal = result.get("abnormal_findings") or []
        if abnormal:
            st.dataframe(abnormal, use_container_width=True, hide_index=True)
        else:
            st.write("No abnormal findings were listed by the model.")

    with tabs[2]:
        st.subheader("Parameter-wise explanations")
        explanations = result.get("parameter_explanations") or []
        if explanations:
            for item in explanations:
                with st.expander(item.get("test_name", "Test explanation")):
                    st.write(item.get("plain_language_explanation", ""))
                    if item.get("possible_significance"):
                        st.markdown("**Possible significance**")
                        st.write(item["possible_significance"])
        else:
            st.write("No explanations were returned.")

    with tabs[3]:
        st.subheader("Report comments and recommendations")
        for item in result.get("report_comments_and_recommendations") or []:
            st.markdown(f"- {item}")

        st.subheader("Follow-up tests mentioned in the report")
        for item in result.get("follow_up_tests_mentioned_in_report") or []:
            st.markdown(f"- {item}")

        st.subheader("Limitations")
        for item in result.get("limitations") or []:
            st.markdown(f"- {item}")

        if result.get("medical_safety_note"):
            st.warning(result["medical_safety_note"])

    with tabs[4]:
        st.json(result)

    st.download_button(
        "Download analysis as JSON",
        data=json.dumps(result, ensure_ascii=False, indent=2),
        file_name="blood_report_analysis.json",
        mime="application/json",
        use_container_width=True,
    )

    st.caption(
        "Medical safety: This tool supports report understanding only. It does not "
        "diagnose disease or replace a clinician's assessment."
    )
