# -*- coding: utf-8 -*-

import os
import pandas as pd
import streamlit as st
import trafilatura
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from typing import Literal


# -----------------------------
# App configuration
# -----------------------------
st.set_page_config(page_title="Agile PM Source Classifier", layout="wide")

st.title("Agile Project Management Source Classifier")
st.write("Classify papers and grey literature into Agile Project Management categories.")


# -----------------------------
# API key configuration
# Works both locally and on Streamlit Cloud
# -----------------------------
load_dotenv()

api_key = st.secrets.get("OPENAI_API_KEY", None) or os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY is missing. Please add it to Streamlit Secrets or your local .env file.")
    st.stop()

client = OpenAI(api_key=api_key)


# -----------------------------
# Categories
# -----------------------------
CATEGORIES = [
    "User Story Management",
    "Backlog Management",
    "Estimation",
    "Task Management",
    "Dependency & Resource Management",
    "Sprint & Project Monitoring",
    "Agile Collaboration Support",
    "Decision Support & Risk Management",
    "Other / To be classified later",
]


class ClassificationResult(BaseModel):
    category: Literal[
        "User Story Management",
        "Backlog Management",
        "Estimation",
        "Task Management",
        "Dependency & Resource Management",
        "Sprint & Project Monitoring",
        "Agile Collaboration Support",
        "Decision Support & Risk Management",
        "Other / To be classified later",
    ]
    subcategory: str
    confidence: Literal["High", "Medium", "Low"]
    reason: str
    evidence: str


SYSTEM_PROMPT = """
You are an expert reviewer conducting a multivocal literature review on
LLM-based multi-agent systems for Agile Project Management.

Your task is to classify each source into exactly one category.

Use the following taxonomy:

1. User Story Management:
   refinement, quality improvement, prioritization of user stories.

2. Backlog Management:
   backlog grooming, backlog organization, backlog prioritization.

3. Estimation:
   user story estimation, story point estimation, effort estimation,
   task effort/time estimation, complexity estimation.

4. Task Management:
   task decomposition, task planning, task scheduling, task prioritization.

5. Dependency & Resource Management:
   dependency detection, blocker identification, assignee allocation,
   resource allocation, role allocation, capability matching.

6. Sprint & Project Monitoring:
   sprint planning, progress tracking, issue tracking, status reporting.

7. Agile Collaboration Support:
   meeting assistance, scrum support, daily scrum, retrospective support,
   collaboration assistance.

8. Decision Support & Risk Management:
   risk prediction, recommendations, quality support, managerial or technical decision support.

9. Other / To be classified later:
   use only if the source does not clearly fit any of the above categories.

Important rules:
- Choose only one main category.
- Do not classify general project management unless it is clearly related to Agile software project management.
- If the source is about user story generation only, classify it as User Story Management only if it also includes refinement, prioritization, or quality improvement.
- If the source is about breaking user stories or requirements into executable tasks, classify it as Task Management.
- If the source is mainly about story points, effort, time, or complexity, classify it as Estimation.
- If uncertain, use "Other / To be classified later" and explain why.
- Keep the reason short and evidence directly grounded in the text.
"""


# -----------------------------
# Functions
# -----------------------------
def classify_text(title: str, text: str) -> ClassificationResult:
    user_prompt = f"""
Source title:
{title}

Source text:
{text[:12000]}

Classify this source according to the taxonomy.
"""

    response = client.responses.parse(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        text_format=ClassificationResult,
    )

    return response.output_parsed


def extract_text_from_url(url: str) -> str:
    downloaded = trafilatura.fetch_url(url)

    if downloaded is None:
        return ""

    extracted = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False
    )

    return extracted or ""


def display_result(result: ClassificationResult):
    st.subheader("Classification Result")
    st.write("**Category:**", result.category)
    st.write("**Subcategory:**", result.subcategory)
    st.write("**Confidence:**", result.confidence)
    st.write("**Reason:**", result.reason)
    st.write("**Evidence:**", result.evidence)


# -----------------------------
# Input mode
# -----------------------------
mode = st.radio(
    "Select input type:",
    ["Paper: Title + Abstract", "Grey Literature: URL", "Batch: Excel/CSV"],
)


# -----------------------------
# Mode 1: Paper title + abstract
# -----------------------------
if mode == "Paper: Title + Abstract":
    title = st.text_input("Paper title")
    abstract = st.text_area("Abstract", height=250)

    if st.button("Classify"):
        if not title.strip() or not abstract.strip():
            st.warning("Please enter both title and abstract.")
        else:
            with st.spinner("Classifying..."):
                result = classify_text(title, abstract)

            display_result(result)


# -----------------------------
# Mode 2: Grey literature URL
# -----------------------------
elif mode == "Grey Literature: URL":
    url = st.text_input("Grey literature URL")

    manual_text = st.text_area(
        "Optional: Paste webpage text manually if URL extraction fails or if you want to classify a selected part of the page",
        height=220
    )

    if st.button("Read URL and Classify"):
        if not url.strip() and not manual_text.strip():
            st.warning("Please enter a URL or paste the webpage text manually.")
        else:
            with st.spinner("Preparing text..."):

                if manual_text.strip():
                    text = manual_text.strip()
                    source_title = url if url.strip() else "Manually pasted grey literature text"
                    st.info("Using manually pasted text.")
                else:
                    text = extract_text_from_url(url)
                    source_title = url

            if not text:
                st.error("Could not extract readable text from this URL. Please paste the webpage text manually.")
            else:
                st.success("Text is ready for classification.")

                with st.expander("Preview text used for classification"):
                    st.write(text[:3000])

                with st.spinner("Classifying..."):
                    result = classify_text(source_title, text)

                display_result(result)


# -----------------------------
# Mode 3: Batch Excel/CSV
# -----------------------------
elif mode == "Batch: Excel/CSV":
    uploaded_file = st.file_uploader("Upload Excel or CSV", type=["xlsx", "csv"])

    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.write("Preview:")
        st.dataframe(df.head())

        st.info(
            "Recommended columns: source_id, source_type, title, abstract, url. "
            "For papers, use title + abstract. For grey literature, use url or paste text in abstract."
        )

        if st.button("Classify all rows"):
            results = []

            progress_bar = st.progress(0)
            total_rows = len(df)

            for index, row in df.iterrows():
                source_id = str(row.get("source_id", "")).strip()
                source_type = str(row.get("source_type", "")).strip()
                title = str(row.get("title", "")).strip()
                abstract = str(row.get("abstract", "")).strip()
                url = str(row.get("url", "")).strip()

                # Handle nan strings
                if source_id.lower() == "nan":
                    source_id = ""
                if source_type.lower() == "nan":
                    source_type = ""
                if title.lower() == "nan":
                    title = ""
                if abstract.lower() == "nan":
                    abstract = ""
                if url.lower() == "nan":
                    url = ""

                # Decide text source
                if url:
                    text = extract_text_from_url(url)

                    # Fallback: if URL extraction fails, use abstract/text column
                    if not text and abstract:
                        text = abstract

                    source_title = title if title else url

                else:
                    text = abstract
                    source_title = title if title else source_id

                if not text:
                    results.append({
                        "source_id": source_id,
                        "source_type": source_type,
                        "title": title,
                        "url": url,
                        "predicted_category": "Other / To be classified later",
                        "predicted_subcategory": "",
                        "confidence": "Low",
                        "reason": "No usable text was available.",
                        "evidence": "",
                        "human_decision": "",
                        "notes": "",
                    })

                    progress_bar.progress((index + 1) / total_rows)
                    continue

                try:
                    result = classify_text(source_title, text)

                    results.append({
                        "source_id": source_id,
                        "source_type": source_type,
                        "title": title,
                        "url": url,
                        "predicted_category": result.category,
                        "predicted_subcategory": result.subcategory,
                        "confidence": result.confidence,
                        "reason": result.reason,
                        "evidence": result.evidence,
                        "human_decision": "",
                        "notes": "",
                    })

                except Exception as e:
                    results.append({
                        "source_id": source_id,
                        "source_type": source_type,
                        "title": title,
                        "url": url,
                        "predicted_category": "Other / To be classified later",
                        "predicted_subcategory": "",
                        "confidence": "Low",
                        "reason": f"Classification failed: {e}",
                        "evidence": "",
                        "human_decision": "",
                        "notes": "",
                    })

                progress_bar.progress((index + 1) / total_rows)

            result_df = pd.DataFrame(results)

            st.subheader("Results")
            st.dataframe(result_df)

            csv = result_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download results as CSV",
                data=csv,
                file_name="agile_pm_classification_results.csv",
                mime="text/csv",
            )
