import os
import sys

from unstract.llmwhisperer import LLMWhispererClientV2
from unstract.llmwhisperer.client import LLMWhispererClientException
import tempfile

from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain.output_parsers import PydanticOutputParser

import pandas as pd
from pydantic import create_model

from datetime import datetime

from dotenv import load_dotenv

import streamlit as st
import uuid

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from transformers import AutoTokenizer, AutoModelForCausalLM

from model import Prompt

B_TEXT = "<|begin_of_text|>"
E_TEXT = "<|eot_id|>"
B_HEAD = "<|start_header_id|>"
E_HEAD = "<|end_header_id|>"

def add_row():
    element_id = uuid.uuid4()
    st.session_state["rows"].append(str(element_id))

def remove_row(row_id):
    st.session_state["rows"].remove(str(row_id))

def assemble_prompt(text, role):
    if role == "system":
        msg = f"{B_TEXT}{B_HEAD}{role}{E_HEAD}\n{text}{E_TEXT}"
    elif role == "user":
        msg = f"{B_HEAD}{role}{E_HEAD}\n{text}{E_TEXT}"
    elif role == "assistant":
        msg = f"{B_HEAD}{role}{E_HEAD}\n{text}{E_TEXT}"
    return msg

def set_model_llama(model_path, access_token):
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=access_token)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        token=access_token,
        torch_dtype="auto"
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer

def generate_prompts_llama(model_path, access_token, doc_type, field, datatype):
    llama_model, llama_tokenizer = set_model_llama(model_path, access_token)

    text = assemble_prompt("You are a helpful assistant that provides prompts for an LLM like yourself for extracting a specific field from a specified document.", "system")
    text += assemble_prompt(f"Provide a prompt for instructing an LLM to extract the {field} from a {doc_type} which is expected to have the datatype: {datatype}", "user")
    text += assemble_prompt("The output should only contain the prompt. You SHOULD NOT include any other text in the response.","assistant")
    
    tokens = llama_tokenizer(text, return_tensors="pt")
    gen = llama_model.generate(input_ids=tokens.input_ids, attention_mask=tokens.attention_mask, max_new_tokens=100).cpu()
    cont = llama_tokenizer.decode(gen[:,tokens.input_ids.shape[1]:][0], skip_special_tokens=True)
    result = cont.strip().split("\n")[-1].replace('"',"").replace("```","").replace("`","")
    prompt_result = Prompt(document_type=doc_type, field_name=field, prompt_instructions=result)
    return prompt_result

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

def get_prompts(fields, document_type):
    gen_prompts = []
    parser = PydanticOutputParser(pydantic_object=Prompt)
    prompt = PromptTemplate(
        template="""You are a helpful assistant that provides prompts for an LLM like yourself for extracting a specific field from a specified document. Answer the user query.
                    {format_instructions}
                    {query}""",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    model = ChatGoogleGenerativeAI(model="gemini-pro")
    chain = prompt | model | parser

    for key in fields:
        gen_prompt = chain.invoke({"query": f"""Provide a prompt for instructing an LLM to extract the field: {key}, of datatype: {fields[key]["datatype"]}, from a document of type: {document_type}"""})
        gen_prompts.append(gen_prompt)

    return gen_prompts

def create_prompt_model(fields, document_type, prompts):
    return create_model(document_type, **{p.field_name: (fields[p.field_name]["datatype"], p.prompt_instructions) for p in prompts})

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

    document_object = parser.parse(response.content)

    return document_object

llama_model_path = "/Users/mwalton/Documents/ELTE/PDF data extraction/Llama-3.2-1B-Instruct"

if __name__ == "__main__":
    with st.sidebar:
        google_api_key = st.text_input("Google API Key", key="google_api_key", type="password")
        llmwhisperer_api_key = st.text_input("LLM Whisperer API Key", key="llmwhisperer_api_key", type="password")
        hugging_face_token = st.text_input("Hugging Face access Token", key="hugging_face_token", type="password")

    st.title("📝 File NLP Extractor Pipeline")

    if google_api_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key
    elif os.environ.get('GOOGLE_API_KEY'):
        google_api_key = os.environ.get('GOOGLE_API_KEY')

    if llmwhisperer_api_key:
        os.environ["LLMWHISPERER_API_KEY"] = llmwhisperer_api_key
    elif os.environ.get('LLMWHISPERER_API_KEY'):
        llmwhisperer_api_key = os.environ.get('LLMWHISPERER_API_KEY')

    if hugging_face_token:
        os.environ["HUGGING_FACE_TOKEN"] = hugging_face_token
    elif os.environ.get('HUGGING_FACE_TOKEN'):
        hugging_face_token = os.environ.get('HUGGING_FACE_TOKEN')

    uploaded_file = st.file_uploader("Upload PDF", type=("pdf"))
    document_type = st.text_input("Document description")
    model_selection = st.selectbox("Select model", ("Gemini", "Llama"))
    
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
        if fields_dict:
            st.subheader("Prompts")
            if prompts_btn:
                if len(rows_collection) > 0:
                    if model_selection == "Llama":
                        if hugging_face_token:
                            with st.spinner('Processing...'):
                                gen_prompts = []
                                for key in fields_dict.keys():
                                    gen_prompt = generate_prompts_llama(llama_model_path, hugging_face_token, document_type, key, fields_dict[key]["datatype"])
                                    gen_prompts.append(gen_prompt)
                                st.session_state["prompts"] = gen_prompts
                        else:
                            st.info("Make sure you have added your Hugging Face access token")
                    if model_selection == "Gemini":
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