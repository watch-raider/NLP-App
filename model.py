from pydantic import BaseModel, Field

class Prompt(BaseModel):
    document_type: str = Field(description="type of document the LLM will be exctracting data from")
    field_name: str = Field(description=f"name of the field the LLM is tasked with extracting from the {document_type}")
    prompt_instructions: str = Field(description=f"prompt for instructing an LLM to extract the {field_name} from the {document_type}")