import streamlit as st
import time
from pathlib import Path
from typing import List, Dict, Any
import pdfplumber
import pandas as pd
from openai import OpenAI
from docx import Document
import json
import io
import os
import re


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Sara - Requirement Analysis",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# OPENROUTER CONFIGURATION
# ============================================================

OPENROUTER_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENROUTER_API_KEY:
    st.error(
        "OPENAI_API_KEY environment variable is not configured. "
        "Please add your OpenRouter API key."
    )
    st.stop()


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "liquid/lfm-2.5-2.6b:free"

# IMPORTANT:
# Smaller chunks prevent the model from producing enormous JSON
# responses that get truncated.
PDF_MAX_PAGES = 8

# DOCX chunk size
DOCX_MAX_WORDS = 3000

# Number of retries for failed AI extraction
MAX_RETRIES = 3

# If a response fails, split the chunk further
MIN_SPLIT_CHARS = 3000

# Maximum number of tasks/deliverables returned per AI call.
# This keeps the response small enough for the model.
MAX_TASKS_PER_RESPONSE = 30
MAX_DELIVERABLES_PER_RESPONSE = 20


# ============================================================
# EMPTY RESULT
# ============================================================

def empty_result() -> Dict[str, List[Dict[str, str]]]:
    return {
        "Tasks": [],
        "Deliverables": []
    }


# ============================================================
# SAFE STRING
# ============================================================

def safe_string(value: Any) -> str:

    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    return str(value).strip()


# ============================================================
# FILE READERS
# ============================================================

def read_file(
    file_path: Path | io.BytesIO,
    file_extension: str
) -> str:

    try:

        # --------------------------------------------------------
        # TXT
        # --------------------------------------------------------

        if file_extension.lower() == ".txt":

            if isinstance(file_path, io.BytesIO):
                file_path.seek(0)
                return file_path.read().decode(
                    "utf-8",
                    errors="ignore"
                )

            return file_path.read_text(
                encoding="utf-8",
                errors="ignore"
            )


        # --------------------------------------------------------
        # PDF
        # --------------------------------------------------------

        elif file_extension.lower() == ".pdf":

            full_text = ""

            with pdfplumber.open(file_path) as pdf:

                for page_num, page in enumerate(pdf.pages, 1):

                    # Normal text
                    page_text = page.extract_text() or ""

                    if page_text:
                        full_text += (
                            f"\n[Page {page_num}]\n"
                            f"{page_text}\n"
                        )

                    # Tables
                    try:

                        tables = page.extract_tables()

                    except Exception:
                        tables = []

                    if tables:

                        for table in tables:

                            for row in table or []:

                                if not row:
                                    continue

                                # Safely handle None cells
                                cells = [
                                    safe_string(cell)
                                    for cell in row
                                ]

                                if not any(cells):
                                    continue

                                row_text = " | ".join(
                                    cell
                                    for cell in cells
                                    if cell
                                )

                                if row_text:
                                    full_text += (
                                        f"{row_text}\n"
                                    )


            # Normalize
            combined_text = (
                full_text
                .replace("\r", "")
                .replace("\x00", "")
                .strip()
            )

            return combined_text


        # --------------------------------------------------------
        # DOCX
        # --------------------------------------------------------

        elif file_extension.lower() == ".docx":

            doc = Document(file_path)

            paragraphs = []

            for paragraph in doc.paragraphs:

                text = paragraph.text or ""

                if text.strip():
                    paragraphs.append(text.strip())

            return "\n".join(paragraphs)


        # --------------------------------------------------------
        # UNSUPPORTED
        # --------------------------------------------------------

        else:

            raise ValueError(
                f"Unsupported file type: {file_extension}"
            )


    except Exception as e:

        st.error(
            f"Failed to read file: "
            f"{type(e).__name__}: {e}"
        )

        return ""


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:

    if not text:
        return ""

    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Normalize common dash characters
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("−", "-")

    lines = text.split("\n")

    cleaned_lines = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


# ============================================================
# PDF CHUNKING
# ============================================================

def chunk_pdf(
    file_path: Path | io.BytesIO,
    max_pages: int = PDF_MAX_PAGES
) -> List[str]:

    chunks = []

    try:

        with pdfplumber.open(file_path) as pdf:

            total_pages = len(pdf.pages)

            for start in range(
                0,
                total_pages,
                max_pages
            ):

                end = min(
                    start + max_pages,
                    total_pages
                )

                chunk_text_parts = []

                for page_number in range(
                    start,
                    end
                ):

                    page = pdf.pages[page_number]

                    page_text = (
                        page.extract_text()
                        or ""
                    )

                    if page_text.strip():

                        chunk_text_parts.append(
                            f"[Page {page_number + 1}]\n"
                            f"{page_text}"
                        )

                    # Extract tables
                    try:
                        tables = page.extract_tables()
                    except Exception:
                        tables = []

                    if tables:

                        for table in tables:

                            for row in table or []:

                                if not row:
                                    continue

                                cells = [
                                    safe_string(cell)
                                    for cell in row
                                ]

                                if not any(cells):
                                    continue

                                row_text = " | ".join(
                                    cell
                                    for cell in cells
                                    if cell
                                )

                                if row_text:
                                    chunk_text_parts.append(
                                        row_text
                                    )

                chunk_text = "\n".join(
                    chunk_text_parts
                )

                chunk_text = clean_text(
                    chunk_text
                )

                if chunk_text:
                    chunks.append(chunk_text)


    except Exception as e:

        st.error(
            f"Error chunking PDF: "
            f"{type(e).__name__}: {e}"
        )

    return chunks


# ============================================================
# DOCX CHUNKING
# ============================================================

def chunk_docx(
    file_path: Path | io.BytesIO,
    max_words: int = DOCX_MAX_WORDS
) -> List[str]:

    chunks = []

    current_paragraphs = []
    current_words = 0

    try:

        doc = Document(file_path)

        for paragraph in doc.paragraphs:

            paragraph_text = (
                paragraph.text or ""
            ).strip()

            if not paragraph_text:
                continue

            words = paragraph_text.split()

            if (
                current_words + len(words)
                > max_words
            ):

                if current_paragraphs:

                    chunks.append(
                        clean_text(
                            "\n".join(
                                current_paragraphs
                            )
                        )
                    )

                current_paragraphs = []
                current_words = 0

            current_paragraphs.append(
                paragraph_text
            )

            current_words += len(words)

        if current_paragraphs:

            chunks.append(
                clean_text(
                    "\n".join(
                        current_paragraphs
                    )
                )
            )


    except Exception as e:

        st.error(
            f"Error chunking DOCX: "
            f"{type(e).__name__}: {e}"
        )

    return chunks


# ============================================================
# REMOVE MARKDOWN / CODE FENCES
# ============================================================

def clean_ai_content(content: Any) -> str:

    if content is None:
        return ""

    # Handle list-based content returned by some models
    if isinstance(content, list):

        parts = []

        for item in content:

            if isinstance(item, dict):

                if "text" in item:
                    parts.append(
                        safe_string(item["text"])
                    )

                elif item.get("type") == "text":
                    parts.append(
                        safe_string(
                            item.get("text", "")
                        )
                    )

            else:
                parts.append(
                    safe_string(item)
                )

        content = "\n".join(parts)

    content = safe_string(content)

    if not content:
        return ""

    # Remove Markdown fences
    content = re.sub(
        r"^\s*```(?:json|python)?\s*",
        "",
        content,
        flags=re.IGNORECASE
    )

    content = re.sub(
        r"\s*```\s*$",
        "",
        content
    )

    # Remove accidental leading text before JSON
    first_brace = content.find("{")

    if first_brace > 0:
        content = content[first_brace:]

    return content.strip()


# ============================================================
# EXTRACT COMPLETE JSON OBJECT
# ============================================================

def extract_json_object(content: str) -> str:

    content = clean_ai_content(content)

    if not content:
        return ""

    start = content.find("{")

    if start == -1:
        return ""

    # Properly track:
    # - braces
    # - strings
    # - escaped quotes
    #
    # This is much safer than simply counting braces.

    depth = 0
    in_string = False
    escape = False

    for i in range(
        start,
        len(content)
    ):

        char = content[i]

        if escape:

            escape = False
            continue

        if char == "\\" and in_string:

            escape = True
            continue

        if char == '"':

            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":

            depth += 1

        elif char == "}":

            depth -= 1

            if depth == 0:

                return content[
                    start:i + 1
                ]

    # JSON was probably truncated
    return ""


# ============================================================
# NORMALIZE RESULT
# ============================================================

def validate_result(
    result: Any
) -> Dict[str, List[Dict[str, str]]]:

    if not isinstance(result, dict):
        return empty_result()

    tasks = result.get(
        "Tasks",
        []
    )

    deliverables = result.get(
        "Deliverables",
        []
    )

    if not isinstance(tasks, list):
        tasks = []

    if not isinstance(
        deliverables,
        list
    ):
        deliverables = []


    # --------------------------------------------------------
    # Normalize Tasks
    # --------------------------------------------------------

    cleaned_tasks = []

    for task in tasks:

        if not isinstance(task, dict):
            continue

        cleaned_task = {
            "Task": safe_string(
                task.get("Task", "")
            ),
            "Parent Task": safe_string(
                task.get("Parent Task", "")
            ),
            "Methodology": safe_string(
                task.get("Methodology", "")
            ),
            "Tools & Technologies": safe_string(
                task.get(
                    "Tools & Technologies",
                    ""
                )
            ),
            "Task Summary": safe_string(
                task.get("Task Summary", "")
            )
        }

        # Don't add completely empty rows
        if not (
            cleaned_task["Task"]
            or cleaned_task["Parent Task"]
        ):
            continue

        # Remove accidental placeholder
        if (
            cleaned_task["Parent Task"]
            .lower()
            == "unspecified task"
        ):
            continue

        cleaned_tasks.append(
            cleaned_task
        )


    # --------------------------------------------------------
    # Normalize Deliverables
    # --------------------------------------------------------

    cleaned_deliverables = []

    for deliverable in deliverables:

        if not isinstance(
            deliverable,
            dict
        ):
            continue

        cleaned_deliverable = {
            "Deliverable": safe_string(
                deliverable.get(
                    "Deliverable",
                    ""
                )
            ),
            "Parent Task": safe_string(
                deliverable.get(
                    "Parent Task",
                    ""
                )
            ),
            "Description": safe_string(
                deliverable.get(
                    "Description",
                    ""
                )
            )
        }

        if not (
            cleaned_deliverable[
                "Deliverable"
            ]
            or cleaned_deliverable[
                "Parent Task"
            ]
        ):
            continue

        if (
            cleaned_deliverable[
                "Parent Task"
            ].lower()
            == "unspecified task"
        ):
            continue

        cleaned_deliverables.append(
            cleaned_deliverable
        )


    return {
        "Tasks": cleaned_tasks,
        "Deliverables": cleaned_deliverables
    }


# ============================================================
# PARSE AI JSON
# ============================================================

def parse_ai_json(
    content: Any
) -> Dict[str, List[Dict[str, str]]]:

    cleaned_content = clean_ai_content(
        content
    )

    if not cleaned_content:
        return empty_result()

    # First attempt: entire response
    try:

        result = json.loads(
            cleaned_content
        )

        return validate_result(
            result
        )

    except json.JSONDecodeError:
        pass


    # Second attempt: locate complete object
    json_content = extract_json_object(
        cleaned_content
    )

    if not json_content:
        return empty_result()

    try:

        result = json.loads(
            json_content
        )

        return validate_result(
            result
        )

    except json.JSONDecodeError:
        return empty_result()


# ============================================================
# AI PROMPT
# ============================================================

def build_prompt(text: str) -> str:

    return f"""
You are an expert federal proposal analyst.

Analyze ONLY the solicitation text provided below.

Extract technical tasks, compliance activities, and explicit deliverables.

IMPORTANT:
Return ONLY valid JSON.
Do not return Markdown.
Do not use ```json.
Do not return Python.
Do not explain anything.

Keep the response concise.

Use this exact JSON structure:

{{
  "Tasks": [
    {{
      "Task": "task name",
      "Parent Task": "exact major task or section heading from source",
      "Methodology": "methodology explicitly stated in source, otherwise empty string",
      "Tools & Technologies": "only tools, technologies, standards, platforms, or frameworks supported by source",
      "Task Summary": "brief source-based summary"
    }}
  ],
  "Deliverables": [
    {{
      "Deliverable": "deliverable name",
      "Parent Task": "exact major task or section heading from source",
      "Description": "brief source-based description"
    }}
  ]
}}

RULES:

1. Do not invent requirements.
2. Do not invent technologies.
3. Do not invent deliverables.
4. Preserve source terminology.
5. Use exact major section/task headings when available.
6. Capture explicit compliance requirements.
7. Capture explicit technical standards.
8. Capture explicit tools and technologies.
9. If methodology is not stated, use an empty string.
10. Every field must contain a string.
11. Never use null.
12. Keep Task Summary concise.
13. Do not duplicate the same task unnecessarily.
14. Do not include commentary outside the JSON.
15. Extract only information supported by the provided text.

Limit the response to approximately:
- 30 tasks maximum
- 20 deliverables maximum

SOLICITATION TEXT:

{text}
"""


# ============================================================
# AI CALL
# ============================================================

def call_ai(
    text: str
) -> Dict[str, List[Dict[str, str]]]:

    if not text or not text.strip():
        return empty_result()


    prompt = build_prompt(
        text
    )


    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            # ------------------------------------------------
            # First attempt with JSON response format
            # ------------------------------------------------

            try:

                response = client.chat.completions.create(

                    model=MODEL_NAME,

                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Return only valid JSON. "
                                "Never return Markdown."
                            )
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],

                    temperature=0.0,

                    # Prevent extremely large responses
                    max_tokens=7000,

                    # Ask OpenRouter/model for JSON
                    response_format={
                        "type": "json_object"
                    }
                )


            except Exception:

                # Some free models/providers may not support
                # response_format. Retry without it.

                response = client.chat.completions.create(

                    model=MODEL_NAME,

                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Return only valid JSON. "
                                "Never return Markdown."
                            )
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],

                    temperature=0.0,

                    max_tokens=7000
                )


            # ------------------------------------------------
            # Validate choices
            # ------------------------------------------------

            if not response.choices:

                st.warning(
                    f"AI returned no choices "
                    f"(attempt {attempt}/{MAX_RETRIES})."
                )

                continue


            # ------------------------------------------------
            # Get message
            # ------------------------------------------------

            message = response.choices[0].message


            # ------------------------------------------------
            # Get content safely
            # ------------------------------------------------

            content = getattr(
                message,
                "content",
                None
            )


            # ------------------------------------------------
            # Handle None
            # ------------------------------------------------

            if content is None:

                st.warning(
                    f"AI returned no final content "
                    f"(attempt {attempt}/{MAX_RETRIES})."
                )

                # Check for refusal
                refusal = getattr(
                    message,
                    "refusal",
                    None
                )

                if refusal:
                    st.warning(
                        f"Model refusal: {refusal}"
                    )

                continue


            content = clean_ai_content(
                content
            )


            if not content:

                st.warning(
                    f"AI returned empty content "
                    f"(attempt {attempt}/{MAX_RETRIES})."
                )

                continue


            # ------------------------------------------------
            # Parse JSON
            # ------------------------------------------------

            result = parse_ai_json(
                content
            )


            if (
                result["Tasks"]
                or result["Deliverables"]
            ):

                return result


            # ------------------------------------------------
            # JSON parsing failed
            # ------------------------------------------------

            st.warning(
                f"Could not parse AI JSON "
                f"(attempt {attempt}/{MAX_RETRIES})."
            )

            # Show truncated diagnostic only
            if attempt == MAX_RETRIES:

                st.code(
                    content[:5000],
                    language="text"
                )


        except Exception as e:

            st.warning(
                f"AI extraction attempt "
                f"{attempt}/{MAX_RETRIES} failed: "
                f"{type(e).__name__}: {e}"
            )

            # Small delay before retry
            time.sleep(
                1.5 * attempt
            )


    return empty_result()


# ============================================================
# RECURSIVE CHUNK EXTRACTION
# ============================================================

def extract_task(
    text: str,
    depth: int = 0
) -> Dict[str, List[Dict[str, str]]]:

    if not text or not text.strip():
        return empty_result()


    # --------------------------------------------------------
    # Try normal extraction
    # --------------------------------------------------------

    result = call_ai(
        text
    )


    if (
        result["Tasks"]
        or result["Deliverables"]
    ):
        return result


    # --------------------------------------------------------
    # If extraction fails, split the chunk
    # --------------------------------------------------------

    if len(text) <= MIN_SPLIT_CHARS:

        st.warning(
            "Unable to extract structured JSON "
            "from this small chunk."
        )

        return empty_result()


    # Prevent infinite recursion
    if depth >= 3:

        st.warning(
            "Maximum chunk splitting depth reached."
        )

        return empty_result()


    st.info(
        "AI response could not be parsed. "
        "Splitting this chunk into smaller sections "
        "and retrying..."
    )


    # --------------------------------------------------------
    # Split by paragraphs where possible
    # --------------------------------------------------------

    paragraphs = [
        p.strip()
        for p in re.split(
            r"\n\s*\n",
            text
        )
        if p.strip()
    ]


    if len(paragraphs) < 2:

        midpoint = len(text) // 2

        parts = [
            text[:midpoint],
            text[midpoint:]
        ]

    else:

        midpoint = len(paragraphs) // 2

        parts = [
            "\n\n".join(
                paragraphs[:midpoint]
            ),
            "\n\n".join(
                paragraphs[midpoint:]
            )
        ]


    combined = empty_result()


    for part_number, part in enumerate(
        parts,
        1
    ):

        st.write(
            f"Retrying smaller section "
            f"{part_number}/{len(parts)}..."
        )

        part_result = extract_task(
            part,
            depth + 1
        )

        combined["Tasks"].extend(
            part_result["Tasks"]
        )

        combined["Deliverables"].extend(
            part_result["Deliverables"]
        )


    return combined


# ============================================================
# DEDUPLICATION
# ============================================================

def consolidate_results(
    results: Dict[str, List[Dict[str, str]]]
) -> Dict[str, List[Dict[str, str]]]:

    # --------------------------------------------------------
    # Tasks
    # --------------------------------------------------------

    task_map = {}

    for task in results.get(
        "Tasks",
        []
    ):

        if not isinstance(task, dict):
            continue

        task_name = safe_string(
            task.get("Task", "")
        )

        parent = safe_string(
            task.get("Parent Task", "")
        )

        if not task_name:
            continue

        key = (
            task_name.lower(),
            parent.lower()
        )

        if key not in task_map:

            task_map[key] = {
                "Task": task_name,
                "Parent Task": parent,
                "Methodology": safe_string(
                    task.get(
                        "Methodology",
                        ""
                    )
                ),
                "Tools & Technologies": safe_string(
                    task.get(
                        "Tools & Technologies",
                        ""
                    )
                ),
                "Task Summary": safe_string(
                    task.get(
                        "Task Summary",
                        ""
                    )
                )
            }

        else:

            existing = task_map[key]

            # Fill missing fields rather than
            # creating duplicate rows.

            for field in [
                "Methodology",
                "Tools & Technologies",
                "Task Summary"
            ]:

                if (
                    not existing[field]
                    and safe_string(
                        task.get(field, "")
                    )
                ):

                    existing[field] = safe_string(
                        task.get(field, "")
                    )


    # --------------------------------------------------------
    # Deliverables
    # --------------------------------------------------------

    deliverable_map = {}

    for deliverable in results.get(
        "Deliverables",
        []
    ):

        if not isinstance(
            deliverable,
            dict
        ):
            continue

        name = safe_string(
            deliverable.get(
                "Deliverable",
                ""
            )
        )

        parent = safe_string(
            deliverable.get(
                "Parent Task",
                ""
            )
        )

        if not name:
            continue

        key = (
            name.lower(),
            parent.lower()
        )

        if key not in deliverable_map:

            deliverable_map[key] = {
                "Deliverable": name,
                "Parent Task": parent,
                "Description": safe_string(
                    deliverable.get(
                        "Description",
                        ""
                    )
                )
            }

        else:

            existing_description = (
                deliverable_map[key][
                    "Description"
                ]
            )

            new_description = safe_string(
                deliverable.get(
                    "Description",
                    ""
                )
            )

            if (
                new_description
                and new_description
                not in existing_description
            ):

                if existing_description:

                    deliverable_map[key][
                        "Description"
                    ] = (
                        existing_description
                        + " "
                        + new_description
                    )

                else:

                    deliverable_map[key][
                        "Description"
                    ] = new_description


    return {
        "Tasks": list(
            task_map.values()
        ),
        "Deliverables": list(
            deliverable_map.values()
        )
    }


# ============================================================
# PROCESS FILE
# ============================================================

def process_file(
    file_path: Path | io.BytesIO,
    file_extension: str
) -> Dict[str, List[Dict[str, str]]]:

    extracted_results = empty_result()


    # --------------------------------------------------------
    # Read file
    # --------------------------------------------------------

    full_text = read_file(
        file_path,
        file_extension
    )

    full_text = clean_text(
        full_text
    )


    if not full_text:

        st.error(
            "No text could be extracted "
            "from this file."
        )

        return extracted_results


    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    if file_extension.lower() == ".pdf":

        try:

            with pdfplumber.open(
                file_path
            ) as pdf:

                total_pages = len(
                    pdf.pages
                )

        except Exception as e:

            st.error(
                f"Could not inspect PDF: "
                f"{type(e).__name__}: {e}"
            )

            return extracted_results


        # ALWAYS use smaller chunks for PDFs
        chunks = chunk_pdf(
            file_path,
            max_pages=PDF_MAX_PAGES
        )

        if total_pages > PDF_MAX_PAGES:

            st.info(
                f"PDF contains {total_pages} pages. "
                f"Using {PDF_MAX_PAGES}-page chunks "
                f"to prevent AI response truncation."
            )


    # --------------------------------------------------------
    # DOCX
    # --------------------------------------------------------

    elif file_extension.lower() == ".docx":

        chunks = chunk_docx(
            file_path,
            max_words=DOCX_MAX_WORDS
        )


    # --------------------------------------------------------
    # TXT
    # --------------------------------------------------------

    else:

        # Split very large TXT files too
        if len(full_text) > 30000:

            paragraphs = [
                p.strip()
                for p in re.split(
                    r"\n\s*\n",
                    full_text
                )
                if p.strip()
            ]

            chunks = []

            current = []
            current_length = 0

            for paragraph in paragraphs:

                if (
                    current_length
                    + len(paragraph)
                    > 25000
                    and current
                ):

                    chunks.append(
                        "\n\n".join(current)
                    )

                    current = []
                    current_length = 0

                current.append(
                    paragraph
                )

                current_length += len(
                    paragraph
                )

            if current:

                chunks.append(
                    "\n\n".join(current)
                )

        else:

            chunks = [
                full_text
            ]


    # --------------------------------------------------------
    # Remove empty chunks
    # --------------------------------------------------------

    chunks = [
        chunk
        for chunk in chunks
        if chunk
        and chunk.strip()
    ]


    if not chunks:

        st.warning(
            "No usable text chunks were "
            "created from the document."
        )

        return extracted_results


    # --------------------------------------------------------
    # Process chunks
    # --------------------------------------------------------

    st.write(
        f"Created {len(chunks)} extraction chunk(s)."
    )


    for i, chunk in enumerate(
        chunks,
        1
    ):

        st.info(
            f"Extracting chunk "
            f"{i}/{len(chunks)}..."
        )

        chunk_result = extract_task(
            chunk
        )

        extracted_results[
            "Tasks"
        ].extend(
            chunk_result.get(
                "Tasks",
                []
            )
        )

        extracted_results[
            "Deliverables"
        ].extend(
            chunk_result.get(
                "Deliverables",
                []
            )
        )


    # --------------------------------------------------------
    # Consolidate
    # --------------------------------------------------------

    extracted_results = consolidate_results(
        extracted_results
    )


    return extracted_results


# ============================================================
# EXCEL CREATION
# ============================================================

def create_excel(
    data: List[Dict[str, Any]]
) -> io.BytesIO:

    output = io.BytesIO()

    if not data:

        df = pd.DataFrame()

    else:

        df = pd.DataFrame(
            data
        )

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        df.to_excel(
            writer,
            index=False,
            sheet_name="Results"
        )

        worksheet = writer.sheets[
            "Results"
        ]

        # Freeze header row
        worksheet.freeze_panes = "A2"

        # Autofilter
        if not df.empty:

            worksheet.auto_filter.ref = (
                worksheet.dimensions
            )

        # Set readable column widths
        for column_cells in worksheet.columns:

            max_length = 0

            column_letter = (
                column_cells[0].column_letter
            )

            for cell in column_cells:

                try:

                    cell_length = len(
                        str(cell.value)
                    )

                    if cell_length > max_length:
                        max_length = cell_length

                except Exception:
                    pass

            worksheet.column_dimensions[
                column_letter
            ].width = min(
                max(max_length + 2, 12),
                60
            )


    output.seek(0)

    return output


# ============================================================
# STREAMLIT APP
# ============================================================

st.title(
    "Sara: Software Automation for Requirement Analysis"
)

st.write(
    "Upload one or more solicitation documents "
    "(TXT, PDF, or DOCX) to extract tasks and deliverables."
)


# ============================================================
# SETTINGS
# ============================================================

with st.expander(
    "Extraction Settings",
    expanded=False
):

    st.write(
        f"**Model:** `{MODEL_NAME}`"
    )

    st.write(
        f"**PDF chunk size:** "
        f"{PDF_MAX_PAGES} pages"
    )

    st.write(
        f"**Maximum retries:** "
        f"{MAX_RETRIES}"
    )

    st.write(
        "Smaller chunks are intentionally used to "
        "prevent truncated JSON responses."
    )


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_files = st.file_uploader(
    "Choose files",
    type=[
        "txt",
        "pdf",
        "docx"
    ],
    accept_multiple_files=True
)


# ============================================================
# PROCESS UPLOADED FILES
# ============================================================

if uploaded_files:

    temp_dir = Path(
        "temp"
    )

    temp_dir.mkdir(
        exist_ok=True
    )


    all_tasks = []
    all_deliverables = []


    start_time = time.time()


    # ========================================================
    # PROCESS EACH FILE
    # ========================================================

    for uploaded_file in uploaded_files:

        st.divider()

        st.write(
            f"## Processing "
            f"{uploaded_file.name}"
        )


        # ----------------------------------------------------
        # Extension
        # ----------------------------------------------------

        file_extension = (
            Path(
                uploaded_file.name
            ).suffix.lower()
        )


        # ----------------------------------------------------
        # Temporary path
        # ----------------------------------------------------

        temp_file_path = (
            temp_dir
            / uploaded_file.name
        )


        # ----------------------------------------------------
        # Save upload
        # ----------------------------------------------------

        try:

            with open(
                temp_file_path,
                "wb"
            ) as f:

                f.write(
                    uploaded_file.getbuffer()
                )

        except Exception as e:

            st.error(
                f"Could not save "
                f"{uploaded_file.name}: "
                f"{type(e).__name__}: {e}"
            )

            continue


        # ----------------------------------------------------
        # Process
        # ----------------------------------------------------

        try:

            extracted = process_file(
                temp_file_path,
                file_extension
            )

        except Exception as e:

            st.error(
                f"Unexpected processing error "
                f"for {uploaded_file.name}: "
                f"{type(e).__name__}: {e}"
            )

            extracted = empty_result()


        # ----------------------------------------------------
        # Add source file
        # ----------------------------------------------------

        for task in extracted.get(
            "Tasks",
            []
        ):

            if isinstance(
                task,
                dict
            ):

                task[
                    "Source File"
                ] = uploaded_file.name


        for deliverable in extracted.get(
            "Deliverables",
            []
        ):

            if isinstance(
                deliverable,
                dict
            ):

                deliverable[
                    "Source File"
                ] = uploaded_file.name


        # ----------------------------------------------------
        # Aggregate
        # ----------------------------------------------------

        all_tasks.extend(
            extracted.get(
                "Tasks",
                []
            )
        )

        all_deliverables.extend(
            extracted.get(
                "Deliverables",
                []
            )
        )


        # ====================================================
        # DISPLAY RESULTS
        # ====================================================

        st.subheader(
            f"Results for "
            f"{uploaded_file.name}"
        )


        # ----------------------------------------------------
        # Tasks
        # ----------------------------------------------------

        if extracted.get(
            "Tasks"
        ):

            st.write(
                "### Extracted Tasks"
            )

            tasks_df = pd.DataFrame(
                extracted["Tasks"]
            )

            st.dataframe(
                tasks_df,
                use_container_width=True,
                hide_index=True
            )


            tasks_excel = create_excel(
                extracted["Tasks"]
            )


            st.download_button(
                label=(
                    f"Download Tasks for "
                    f"{uploaded_file.name} "
                    f"as Excel"
                ),
                data=tasks_excel,
                file_name=(
                    f"{Path(uploaded_file.name).stem}"
                    "_tasks.xlsx"
                ),
                mime=(
                    "application/vnd.openxmlformats-"
                    "officedocument.spreadsheetml.sheet"
                ),
                key=(
                    f"tasks_"
                    f"{uploaded_file.name}"
                )
            )

        else:

            st.warning(
                f"No tasks extracted from "
                f"{uploaded_file.name}."
            )


        # ----------------------------------------------------
        # Deliverables
        # ----------------------------------------------------

        if extracted.get(
            "Deliverables"
        ):

            st.write(
                "### Extracted Deliverables"
            )

            deliverables_df = pd.DataFrame(
                extracted["Deliverables"]
            )

            st.dataframe(
                deliverables_df,
                use_container_width=True,
                hide_index=True
            )


            deliverables_excel = create_excel(
                extracted[
                    "Deliverables"
                ]
            )


            st.download_button(
                label=(
                    f"Download Deliverables for "
                    f"{uploaded_file.name} "
                    f"as Excel"
                ),
                data=deliverables_excel,
                file_name=(
                    f"{Path(uploaded_file.name).stem}"
                    "_deliverables.xlsx"
                ),
                mime=(
                    "application/vnd.openxmlformats-"
                    "officedocument.spreadsheetml.sheet"
                ),
                key=(
                    f"deliverables_"
                    f"{uploaded_file.name}"
                )
            )


        else:

            st.warning(
                f"No deliverables extracted from "
                f"{uploaded_file.name}."
            )


        # ----------------------------------------------------
        # Cleanup
        # ----------------------------------------------------

        try:

            os.remove(
                temp_file_path
            )

        except OSError:
            pass


    # ========================================================
    # AGGREGATED RESULTS
    # ========================================================

    if (
        all_tasks
        or all_deliverables
    ):

        st.divider()

        st.header(
            "Aggregated Results Across All Files"
        )


        # ----------------------------------------------------
        # All Tasks
        # ----------------------------------------------------

        if all_tasks:

            st.subheader(
                "All Extracted Tasks"
            )

            all_tasks_df = pd.DataFrame(
                all_tasks
            )

            st.dataframe(
                all_tasks_df,
                use_container_width=True,
                hide_index=True
            )


            all_tasks_excel = create_excel(
                all_tasks
            )


            st.download_button(
                label=(
                    "Download All Tasks as Excel"
                ),
                data=all_tasks_excel,
                file_name="all_tasks.xlsx",
                mime=(
                    "application/vnd.openxmlformats-"
                    "officedocument.spreadsheetml.sheet"
                ),
                key="all_tasks_download"
            )


        # ----------------------------------------------------
        # All Deliverables
        # ----------------------------------------------------

        if all_deliverables:

            st.subheader(
                "All Extracted Deliverables"
            )

            all_deliverables_df = pd.DataFrame(
                all_deliverables
            )

            st.dataframe(
                all_deliverables_df,
                use_container_width=True,
                hide_index=True
            )


            all_deliverables_excel = create_excel(
                all_deliverables
            )


            st.download_button(
                label=(
                    "Download All Deliverables "
                    "as Excel"
                ),
                data=all_deliverables_excel,
                file_name="all_deliverables.xlsx",
                mime=(
                    "application/vnd.openxmlformats-"
                    "officedocument.spreadsheetml.sheet"
                ),
                key="all_deliverables_download"
            )


    # ========================================================
    # PROCESSING TIME
    # ========================================================

    elapsed = round(
        time.time() - start_time,
        2
    )

    st.success(
        f"Finished processing "
        f"{len(uploaded_files)} file(s) "
        f"in {elapsed} seconds."
    )


else:

    st.info(
        "Please upload one or more files "
        "to begin processing."
    )
