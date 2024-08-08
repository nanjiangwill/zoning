import os

import streamlit as st

from zoning.district_extraction.__main__ import main as district_extraction_main
from zoning.eval.__main__ import main as eval_main
from zoning.format_ocr.__main__ import main as format_ocr_main
from zoning.index.__main__ import main as index_main
from zoning.llm.__main__ import main as llm_main
from zoning.normalization.__main__ import main as normalization_main
from zoning.ocr.__main__ import main as ocr_main
from zoning.prompt.__main__ import main as prompt_main
from zoning.search.__main__ import main as search_main

data_path = "data"  # sys.argv[1]

st.title("Zoning User Mode")

st.subheader("How to use Zoning User Mode")
st.write("1. Upload the zoning files.")
st.write("2. Run the OCR pipeline.")
st.write("3. Run the District Extraction pipeline.")
st.write("4. Run the Zoning Pipeline.")
st.write("5. View the results.")


def mkdir_for_new_state_and_store_pdfs(state, files):
    if not state:
        return st.warning("Please enter a state name")

    formatted_state = state.replace(" ", "_").lower()
    state_dir = os.path.join(data_path, formatted_state)
    pdf_dir = os.path.join(state_dir, "pdfs")

    os.makedirs(pdf_dir, exist_ok=True)

    for file in files:
        file_path = os.path.join(pdf_dir, file.name)
        if os.path.exists(file_path):
            st.warning(f"File already exists: {file.name}")
        else:
            with open(file_path, "wb") as f:
                f.write(file.getvalue())
            st.success(f"File saved: {file.name}")


state = st.text_input(
    label="Enter the state of the zoning files",
    placeholder="Enter the state of the zoning files",
)

uploaded_file = st.file_uploader(
    label="Upload Zoning Files",
    accept_multiple_files=True,
)
if uploaded_file is not None:
    files = uploaded_file
    print(files)

mode = st.radio(
    "Which pipeline would you like to run?",
    options=["OCR", "District Extraction", "Zoning Pipeline"],
)


if st.button("Run Pipeline"):
    if mode == "OCR":
        mkdir_for_new_state_and_store_pdfs(state, files)
        format_ocr_main(files)
    elif mode == "District Extraction":
        district_extraction_main(files)
    elif mode == "Zoning Pipeline":
        index_main(files)
        search_main(files)
        prompt_main(files)
        llm_main(files)
        normalization_main(files)
        eval_main(files)
        st.write("Pipeline Completed, View Results.")
