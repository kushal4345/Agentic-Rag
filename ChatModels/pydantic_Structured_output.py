from pydantic import BaseModel, Field , EmailStr # emailstr is used to validate the email format
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain.schema.messages import HumanMessage, AIMessage
from typing import Optional

class Review(BaseModel):
    """A class to represent a review."""
    review_id: str = Field(..., description="Unique identifier for the review")
    user_id: str = Field(..., description="Unique identifier for the user")
    product_id: str = Field(..., description="Unique identifier for the product")
    rating: float = Field(..., ge=1.0, le=5.0, description="Rating given by the user, typically between 1 and 5")
    comment: str = Field(None, description="Optional comment about the product")
    comment : Optional[str] = Field(None, description="Optional comment about the product")
    email : EmailStr = Field(..., description="Email address of the user")
    cgpa: float = Field(..., ge=0.0, le=10.0, default=5, description="CGPA of the user, typically between 0.0 and 10.0") # field conatins various things like range description etc
                                 
       #if it present then will give output else give none 

# Example data
person = {   #if i write the differnt type value that i didn't define in the model it will raise an error that is the main use of pydantic
    'review_id': 'rev_001',
    'user_id': '5656',
    'product_id': 'prod_123',
    'rating': 4.5,
    'comment': 'Excellent quality and fast delivery!',
    'email': 'kushal4345@gmai.com' # 'kushaohd' this will raise an error because it is not a valid email format
    
}

# Creating an instance

new_person = Review(**person) 

# Optional: Print it
print(new_person)
