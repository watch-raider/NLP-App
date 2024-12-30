from pydantic import BaseModel, Field

class Prompt(BaseModel):
    document_type: str = Field(description="type of document the AI agent will be exctracting from")
    field_name: str = Field(description="name of the field the AI agent is looking to extract from the field")
    prompt_instructions: str = Field(description=f"prompt for instructing an AI agent to extract the specified field from a document")