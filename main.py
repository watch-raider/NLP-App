import os
import sys

from unstract.llmwhisperer import LLMWhispererClientV2
from unstract.llmwhisperer.client import LLMWhispererClientException
import tempfile

from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser

import pandas as pd
import json
from pydantic import create_model

from datetime import datetime

from dotenv import load_dotenv

from model import Prompt

import streamlit as st
import uuid

def create_prompt_model(fields, document_type, prompts):
    return create_model(document_type, **{p.field_name: (fields[p.field_name]["datatype"], p.prompt_instructions) for p in prompts})

def get_prompts(fields, document_type):
    gen_prompts = []
    for key in fields:
        human_text = "{instruction}\n{format_instructions}"
        message = HumanMessagePromptTemplate.from_template(human_text)
        gen_prompt = ChatPromptTemplate.from_messages([message])

        model = ChatGoogleGenerativeAI(model="gemini-pro")

        chain = gen_prompt | model | output_parser
        gen_prompt = chain.invoke(
            {"instruction":f"""
            You are a helpful assistant that provides prompts for an AI agent like yourself for extracting a specific field from a specified document.
            Can you provide a prompt for instructing an AI agent to extract the field: {key}, of datatype: {fields[key]["datatype"]}, from a document of type: {document_type}
            """,
            "format_instructions":format_instructions})
        gen_prompts.append(gen_prompt)

    return gen_prompts

def add_row():
    element_id = uuid.uuid4()
    st.session_state["rows"].append(str(element_id))

def remove_row(row_id):
    st.session_state["rows"].remove(str(row_id))

def generate_row(row_id):
    row_container = st.empty()
    row_columns = row_container.columns((3, 2, 1))
    row_name = row_columns[0].text_input("Field Name", key=f"txt_{row_id}")
    row_datatype = row_columns[1].selectbox(
        "Select Datatype", 
        ("text", "number", "decimal", "datetime"),
        key=f"datatype_{row_id}"
    )
    row_columns[2].button("🗑️", key=f"del_{row_id}", on_click=remove_row, args=[row_id])
    return {"name": row_name, "datatype": row_datatype}

def extract_txt_from_pdf(file_path):
    client = LLMWhispererClientV2()

    try:
        result = client.whisper(
                    file_path=file_path,
                    wait_for_completion=True,
                    wait_timeout=200,
                )
        return result
    except LLMWhispererClientException as e:
        st.write(e)

def make_llm_chat_call(text, document_type, model):
    preamble = ("\n"
                "Your ability to extract and summarize this information accurately is essential for effective "
                f"{document_type} analysis. Pay close attention to the {document_type}'s language, "
                "structure, and any cross-references to ensure a comprehensive and precise extraction of "
                "information. Do not use prior knowledge or information from outside the context to answer the "
                "questions. Only use the information provided in the context to answer the questions.\n")
    postamble = "Do not include any explanation in the reply. Only include the extracted information in the reply."
    system_template = "{preamble}"
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = "{format_instructions}\n{raw_file_data}\n{postamble}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    parser = PydanticOutputParser(pydantic_object=model)

    # compile chat template
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    request = chat_prompt.format_prompt(preamble=preamble,
                                        format_instructions=parser.get_format_instructions(),
                                        raw_file_data=text,
                                        postamble=postamble).to_messages()

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    response = model.invoke(request)

    invoice_object = parser.parse(response.content)

    return invoice_object


if __name__ == "__main__":
    with st.sidebar:
        google_api_key = st.text_input("Google API Key", key="google_api_key", type="password")
        llmwhisperer_api_key = st.text_input("LLM Whisperer API Key", key="llmwhisperer_api_key", type="password")
        #google_json_file = st.file_uploader("Upload Google Application Credentials", type=("json"))

    st.title("📝 File NLP Extractor Pipeline")

    if google_api_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key
    elif os.environ.get('GOOGLE_API_KEY'):
        google_api_key = os.environ.get('GOOGLE_API_KEY')

    if llmwhisperer_api_key:
        os.environ["LLMWHISPERER_API_KEY"] = llmwhisperer_api_key
    elif os.environ.get('LLMWHISPERER_API_KEY'):
        llmwhisperer_api_key = os.environ.get('LLMWHISPERER_API_KEY')

    uploaded_file = st.file_uploader("Upload PDF", type=("pdf"))
    document_type = st.text_input("Document description")
    
    if uploaded_file:
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())

    load_dotenv()

    fields_dict = {}

    output_parser = PydanticOutputParser(pydantic_object=Prompt)
    format_instructions = output_parser.get_format_instructions()

    if uploaded_file and document_type:
        st.subheader("Data fields")

        if "rows" not in st.session_state:
            st.session_state["rows"] = []

        rows_collection = []

        for row in st.session_state["rows"]:
            row_data = generate_row(row)
            rows_collection.append(row_data)

        menu = st.columns(2)

        with menu[0]:
            st.button("Add Field", on_click=add_row)
        with menu[1]:
            prompts_btn = st.button("Generate prompts")

        if len(rows_collection) > 0:
            data = pd.DataFrame(rows_collection)
            for i, x in data.iterrows():
                if len(x["name"]) > 0:
                    field_name = x["name"].strip().replace(" ", "_").casefold()
                    fields_dict[f"{field_name}"] = {}
                    fields_dict[f"{field_name}"]["friendly_name"] = x["name"]

                    if x["datatype"] == "datetime":
                        fields_dict[f"{field_name}"]["datatype"] = datetime
                    elif x["datatype"] == "number":
                        fields_dict[f"{field_name}"]["datatype"] = int
                    elif x["datatype"] == "decimal":
                        fields_dict[f"{field_name}"]["datatype"] = float
                    else:
                        fields_dict[f"{field_name}"]["datatype"] = str

        # Outputed schema we will pass to the LLM containing the desired fields, instructions on how to extract the data and their expected datatype
        #dynamic_model = get_prompt(fields_dict, "Invoice")
        if fields_dict:
            st.subheader("Prompts")
            if prompts_btn:
                if len(rows_collection) > 0:
                    if google_api_key:
                        with st.spinner('Processing...'):
                            st.session_state["prompts"] = get_prompts(fields_dict, document_type)
                    else:
                        st.info("Make sure you have added your Google API key")
                else:
                    st.info("Please add fields.")

            if "prompts" in st.session_state:
                for i, p in enumerate(st.session_state["prompts"]):
                    st.session_state["prompts"][i].prompt_instructions = st.text_input(f'{fields_dict[f"{p.field_name}"]["friendly_name"]} prompt', p.prompt_instructions, key=f'prompt_{i}') 

                confirm_prompts_btn = st.button("Confirm prompts")

                if confirm_prompts_btn:
                    p_model = create_prompt_model(fields_dict, document_type, st.session_state["prompts"])
                    if llmwhisperer_api_key:
                        st.subheader("Output")
                        pdf_txt = ""
                        with st.spinner('Processing...'):
                            txt_result = extract_txt_from_pdf(file_path)
                            pdf_txt = txt_result["extraction"]["result_text"]
                        if len(pdf_txt) > 0:
                            document_obj = make_llm_chat_call(pdf_txt, document_type, p_model)
                            json_obj = document_obj.model_dump_json()
                            st.json(json_obj)
                    else:
                        st.info("Make sure you have added your LLM Whisperer API key")
    else:
        st.info("Please upload PDF and specify description of document")


        






