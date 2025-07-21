from langchain_openai import AzureChatOpenAI
from langchain.schema.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from typing import Optional , Literal , TypedDict, Annotated

class review(TypedDict):
    review_id: Annotated[ str , "Unique identifier for the review" ]  #Annotated is used to remove the amibguity of the type
    user_id: Annotated[str , "Unique identifier for the user"]
    product_id: Annotated[str , "Unique identifier for the product"]
    rating: Annotated[float , "Rating given by the user, typically between 1 and 5"]
    comment: Optional[str]

load_dotenv()
# Azure Chat OpenAI
llm = AzureChatOpenAI(
    deployment_name="gpt-4o",  # Your Azure deployment name (not model name)
    openai_api_version="2025-01-01-preview",
    temperature=0.7
)
Structured_model = llm.with_structured_output(review)  # it help to identify key points from the output of the model
output = Structured_model.invoke("""Review ID rev_20250719_00123 was submitted by user user_789456123 for the product with ID prod_AZ_1123. The user gave a rating of 4.5 out of 5, expressing their satisfaction with the product. In their comment, they mentioned, "Great product! The build quality is excellent and the delivery was super fast. Will definitely recommend it to friends.
""")
print(output)
print(output['review_id'])

# for data validation we will use pydantic