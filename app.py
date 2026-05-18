# -*- coding: utf-8 -*-

import os
from io import BytesIO
from typing import Literal

import pandas as pd
import streamlit as st
import trafilatura
import plotly.express as px
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel


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
    st.error(
        "OPENAI_API_KEY is missing. Please add it to Streamlit Secrets or your local .env file."
    )
    st.stop()

client = OpenAI(api_key=api_key)


# -----------------------------
# Categories and subcategories
# -----------------------------
MAIN_CATEGORIES = [
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


ALLOWED_SUBCATEGORIES_BY_CATEGORY = {
    "User Story Management": [
        "refinement",
        "quality improvement",
        "prioritization of user stories",
    ],
    "Backlog Management": [
        "backlog grooming",
        "backlog organization",
        "backlog prioritization",
    ],
    "Estimation": [
        "user story estimation",
        "story point estimation",
        "effort estimation",
        "task effort/time estimation",
        "complexity estimation",
    ],
    "Task Management": [
        "task decomposition",
        "task planning",
        "task scheduling",
        "task prioritization",
    ],
    "Dependency & Resource Management": [
        "dependency detection",
        "blocker identification",
        "assignee allocation",
        "resource allocation",
        "role allocation",
        "capability matching",
    ],
    "Sprint & Project Monitoring": [
        "sprint planning",
        "progress tracking",
        "issue tracking",
        "status reporting",
    ],
    "Agile Collaboration Support": [
        "meeting assistance",
        "scrum support",
        "daily scrum",
        "retrospective support",
        "collaboration assistance",
    ],
    "Decision Support & Risk Management": [
        "risk prediction",
        "recommendations",
        "quality support",
        "managerial or technical decision support",
    ],
    "Other / To be classified later": [
        "not clearly classifiable",
        "insufficient information",
        "outside Agile software project management",
    ],
}


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

    subcategories: list[
        Literal[
            "refinement",
            "quality improvement",
            "prioritization of user stories",
            "backlog grooming",
            "backlog organization",
            "backlog prioritization",
            "user story estimation",
            "story point estimation",
            "effort estimation",
            "task effort/time estimation",
            "complexity estimation",
            "task decomposition",
            "task planning",
            "task scheduling",
            "task prioritization",
            "dependency detection",
            "blocker identification",
            "assignee allocation",
            "resource allocation",
            "role allocation",
            "capability matching",
            "sprint planning",
            "progress tracking",
            "issue tracking",
            "status reporting",
            "meeting assistance",
            "scrum support",
            "daily scrum",
            "retrospective support",
            "collaboration assistance",
            "risk prediction",
            "recommendations",
            "quality support",
            "managerial or technical decision support",
            "not clearly classifiable",
            "insufficient information",
            "outside Agile software project management",
        ]
    ]

    confidence: Literal["High", "Medium", "Low"]
    reason: str
    evidence: str


SYSTEM_PROMPT = """
You are an expert reviewer conducting a multivocal literature review on
LLM-based multi-agent systems for Agile Project Management.

Your task is to classify each source into exactly one main category and one or more predefined subcategories.

Use only the taxonomy below.

Taxonomy:

1. Main Category: User Story Management
   Allowed subcategories:
   - refinement
   - quality improvement
   - prioritization of user stories

2. Main Category: Backlog Management
   Allowed subcategories:
   - backlog grooming
   - backlog organization
   - backlog prioritization

3. Main Category: Estimation
   Allowed subcategories:
   - user story estimation
   - story point estimation
   - effort estimation
   - task effort/time estimation
   - complexity estimation

4. Main Category: Task Management
   Allowed subcategories:
   - task decomposition
   - task planning
   - task scheduling
   - task prioritization

5. Main Category: Dependency & Resource Management
   Allowed subcategories:
   - dependency detection
   - blocker identification
   - assignee allocation
   - resource allocation
   - role allocation
   - capability matching

6. Main Category: Sprint & Project Monitoring
   Allowed subcategories:
   - sprint planning
   - progress tracking
   - issue tracking
   - status reporting

7. Main Category: Agile Collaboration Support
   Allowed subcategories:
   - meeting assistance
   - scrum support
   - daily scrum
   - retrospective support
   - collaboration assistance

8. Main Category: Decision Support & Risk Management
   Allowed subcategories:
   - risk prediction
   - recommendations
   - quality support
   - managerial or technical decision support

9. Main Category: Other / To be classified later
   Allowed subcategories:
   - not clearly classifiable
   - insufficient information
   - outside Agile software project management

Important rules:
- Choose exactly one main category.
- Choose one or more subcategories.
- All selected subcategories must belong to the selected main category.
- Do not create new main category names.
- Do not create new subcategory names.
- If the source covers multiple subtopics within the same main category, include all relevant subcategories.
- If the source covers multiple main categories, choose the most central or dominant main category, and select only subcategories from that main category.
- Do not classify general project management unless it is clearly related to Agile software project management.
- If the source is only about user story generation, classify it as User Story Management only if it also includes refinement, quality improvement, or prioritization of user stories.
- If the source is about breaking user stories or requirements into executable tasks, classify it as Task Management.
- If the source is mainly about story points, effort, time, or complexity, classify it as Estimation.
- If the source is too broad but still clearly related to Agile software project management, choose the dominant category.
- If the source does not clearly fit any category, use Other / To be classified later.
- Keep the reason short.
- Evidence must be a short phrase or sentence grounded in the provided source text.
"""


# -----------------------------
# Helper functions
# -----------------------------
def clean_cell(value) -> str:
    value = str(value).strip()

    if value.lower() == "nan":
        return ""

    return value


def clean_subcategories(result: ClassificationResult) -> ClassificationResult:
    """
    Ensures that all returned subcategories belong to the selected main category.
    If the model returns a subcategory from another category, it is removed.
    """

    allowed_subcategories = ALLOWED_SUBCATEGORIES_BY_CATEGORY[result.category]

    cleaned_subcategories = [
        subcategory
        for subcategory in result.subcategories
        if subcategory in allowed_subcategories
    ]

    if not cleaned_subcategories:
        if result.category == "Other / To be classified later":
            cleaned_subcategories = ["insufficient information"]
        else:
            cleaned_subcategories = [allowed_subcategories[0]]

    result.subcategories = cleaned_subcategories
    return result


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

    result = response.output_parsed
    result = clean_subcategories(result)

    return result


def extract_text_from_url(url: str) -> str:
    downloaded = trafilatura.fetch_url(url)

    if downloaded is None:
        return ""

    extracted = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
    )

    return extracted or ""


def display_result(result: ClassificationResult):
    st.subheader("Classification Result")
    st.write("**Category:**", result.category)
    st.write("**Subcategories:**", ", ".join(result.subcategories))
    st.write("**Confidence:**", result.confidence)
    st.write("**Reason:**", result.reason)
    st.write("**Evidence:**", result.evidence)


def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="classification_results")

    return output.getvalue()


def build_main_category_chart(result_df: pd.DataFrame):
    category_counts = (
        result_df["predicted_category"]
        .value_counts()
        .reset_index()
    )
    category_counts.columns = ["Main Category", "Count"]

    fig = px.bar(
        category_counts,
        x="Main Category",
        y="Count",
        title="Distribution of Main Categories",
        text="Count",
    )

    fig.update_traces(textposition="outside")

    fig.update_layout(
        xaxis_title="Main Category",
        yaxis_title="Number of Sources",
        xaxis_tickangle=-30,
        height=520,
        showlegend=False,
    )

    return fig


def build_subcategory_chart(result_df: pd.DataFrame, top_n: int = 20):
    subcategory_series = (
        result_df["predicted_subcategories"]
        .fillna("")
        .astype(str)
        .str.split(",")
        .explode()
        .str.strip()
    )

    subcategory_series = subcategory_series[subcategory_series != ""]

    subcategory_counts = (
        subcategory_series
        .value_counts()
        .head(top_n)
        .reset_index()
    )

    subcategory_counts.columns = ["Subcategory", "Count"]

    fig = px.bar(
        subcategory_counts,
        x="Subcategory",
        y="Count",
        title=f"Top {top_n} Subcategories",
        text="Count",
    )

    fig.update_traces(textposition="outside")

    fig.update_layout(
        xaxis_title="Subcategory",
        yaxis_title="Number of Mentions",
        xaxis_tickangle=-45,
        height=620,
        showlegend=False,
    )

    return fig


def build_confidence_chart(result_df: pd.DataFrame):
    confidence_counts = (
        result_df["confidence"]
        .value_counts()
        .reset_index()
    )

    confidence_counts.columns = ["Confidence", "Count"]

    fig = px.pie(
        confidence_counts,
        names="Confidence",
        values="Count",
        title="Confidence Distribution",
        hole=0.4,
    )

    fig.update_layout(height=450)

    return fig


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
        height=220,
    )

    if st.button("Read URL and Classify"):
        if not url.strip() and not manual_text.strip():
            st.warning("Please enter a URL or paste the webpage text manually.")
        else:
            with st.spinner("Preparing text..."):
                if manual_text.strip():
                    text = manual_text.strip()
                    source_title = url if url.strip() else "Manually pasted grey literature text"
                    text_source_used = "manual_text"
                    st.info("Using manually pasted text.")
                else:
                    text = extract_text_from_url(url)
                    source_title = url
                    text_source_used = "url"

            if not text:
                st.error(
                    "Could not extract readable text from this URL. Please paste the webpage text manually."
                )
            else:
                st.success(f"Text is ready for classification. Source used: {text_source_used}")

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
            "If abstract is available, the app will use abstract first. "
            "If abstract is empty, it will try to extract text from the URL."
        )

        required_columns = {"source_id", "source_type", "title", "abstract", "url"}
        available_columns = set(df.columns)

        missing_columns = required_columns - available_columns
        if missing_columns:
            st.warning(
                "Some recommended columns are missing: "
                + ", ".join(sorted(missing_columns))
                + ". The app can still run, but results may be less complete."
            )

        if st.button("Classify all rows"):
            results = []

            progress_bar = st.progress(0)
            total_rows = len(df)

            for row_number, (_, row) in enumerate(df.iterrows(), start=1):
                source_id = clean_cell(row.get("source_id", ""))
                source_type = clean_cell(row.get("source_type", ""))
                title = clean_cell(row.get("title", ""))
                abstract = clean_cell(row.get("abstract", ""))
                url = clean_cell(row.get("url", ""))

                # Important logic:
                # 1. If abstract exists, use abstract first.
                # 2. If abstract is empty and url exists, extract text from url.
                # 3. If neither exists, classify as insufficient information.
                if abstract:
                    text = abstract
                    source_title = title if title else source_id if source_id else "Untitled source"
                    text_source_used = "abstract"
                elif url:
                    text = extract_text_from_url(url)
                    source_title = title if title else url
                    text_source_used = "url"
                else:
                    text = ""
                    source_title = title if title else source_id if source_id else "Untitled source"
                    text_source_used = "none"

                if not text:
                    results.append(
                        {
                            "source_id": source_id,
                            "source_type": source_type,
                            "title": title,
                            "url": url,
                            "text_source_used": text_source_used,
                            "predicted_category": "Other / To be classified later",
                            "predicted_subcategories": "insufficient information",
                            "confidence": "Low",
                            "reason": "No usable text was available.",
                            "evidence": "",
                            "human_decision": "",
                            "notes": "",
                        }
                    )

                    progress_bar.progress(row_number / total_rows)
                    continue

                try:
                    result = classify_text(source_title, text)

                    results.append(
                        {
                            "source_id": source_id,
                            "source_type": source_type,
                            "title": title,
                            "url": url,
                            "text_source_used": text_source_used,
                            "predicted_category": result.category,
                            "predicted_subcategories": ", ".join(result.subcategories),
                            "confidence": result.confidence,
                            "reason": result.reason,
                            "evidence": result.evidence,
                            "human_decision": "",
                            "notes": "",
                        }
                    )

                except Exception as e:
                    results.append(
                        {
                            "source_id": source_id,
                            "source_type": source_type,
                            "title": title,
                            "url": url,
                            "text_source_used": text_source_used,
                            "predicted_category": "Other / To be classified later",
                            "predicted_subcategories": "insufficient information",
                            "confidence": "Low",
                            "reason": f"Classification failed: {e}",
                            "evidence": "",
                            "human_decision": "",
                            "notes": "",
                        }
                    )

                progress_bar.progress(row_number / total_rows)

            result_df = pd.DataFrame(results)

            st.subheader("Results")
            st.dataframe(result_df)

            # -----------------------------
            # Charts
            # -----------------------------
            st.subheader("Visual Summary")

            col_chart_1, col_chart_2 = st.columns([2, 1])

            with col_chart_1:
                main_category_fig = build_main_category_chart(result_df)
                st.plotly_chart(main_category_fig, use_container_width=True)

            with col_chart_2:
                confidence_fig = build_confidence_chart(result_df)
                st.plotly_chart(confidence_fig, use_container_width=True)

            subcategory_fig = build_subcategory_chart(result_df, top_n=20)
            st.plotly_chart(subcategory_fig, use_container_width=True)

            # -----------------------------
            # Download buttons
            # -----------------------------
            csv = result_df.to_csv(index=False).encode("utf-8")
            excel_bytes = dataframe_to_excel_bytes(result_df)

            col1, col2 = st.columns(2)

            with col1:
                st.download_button(
                    "Download results as CSV",
                    data=csv,
                    file_name="agile_pm_classification_results.csv",
                    mime="text/csv",
                )

            with col2:
                st.download_button(
                    "Download results as Excel",
                    data=excel_bytes,
                    file_name="agile_pm_classification_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
