import json
from langchain_openai import AzureChatOpenAI
from langchain.output_parsers.openai_tools import JsonOutputKeyToolsParser
from dotenv import load_dotenv

load_dotenv()

# Load JSON schema
json_schema = {
  "title": "Review",
  "type": "object",
  "properties": {
    "key_themes": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Write down all the key themes discussed in the review in a list"
    },
    "summary": {
      "type": "string",
      "description": "A brief summary of the review"
    },
    "sentiment": {
      "type": "string",
      "enum": ["positive", "neutral", "negative"],
      "description": "Sentiment of the review"
    }
  },
  "required": ["key_themes", "summary", "sentiment"]
}

# Setup Azure LLM
llm = AzureChatOpenAI(
    deployment_name="gpt-4o",
    openai_api_version="2025-01-01-preview",
    temperature=0.7
)

# Use with structured output
llm_with_schema = llm.with_structured_output(json_schema)

# Review text
text = """
Review ID rev_20250719_00123 was submitted by user user_789456123 for the product with ID prod_AZ_1123.
The user gave a rating of 4.5 out of 5, expressing their satisfaction with the product.
In their comment, they mentioned, "Great product! The build quality is excellent and the delivery was super fast.
Will definitely recommend it to friends."
"""

# Invoke and print result
response = llm_with_schema.invoke(text)
print(response)
