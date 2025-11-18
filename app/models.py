"""
Pydantic models for request/response validation and data structures.
"""
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    userId: str = Field(..., min_length=1, description="Unique user identifier")
    userMessage: str = Field(..., min_length=1, description="User's chat message")
    chatInterest: bool = Field(..., description="Whether this is a first-time interaction")
    interestTopic: Optional[str] = Field(None, description="Topic of interest for first-time users")
    
    @model_validator(mode='after')
    def validate_interest_topic(self):
        """Validate that interestTopic is provided when chatInterest is true."""
        if self.chatInterest and not self.interestTopic:
            raise ValueError('interestTopic is required when chatInterest is true')
        return self


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str = Field(..., description="AI-generated response in Markdown format")


class Message(BaseModel):
    """Individual message in chat history."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    
    @field_validator('role')
    @classmethod
    def validate_role(cls, v):
        """Validate that role is either 'user' or 'assistant'."""
        if v not in ['user', 'assistant']:
            raise ValueError("role must be either 'user' or 'assistant'")
        return v


class UserContext(BaseModel):
    """User context data structure for cache and database."""
    chatHistory: List[Message] = Field(default_factory=list, description="Ordered list of chat messages")
    chatInterest: Optional[str] = Field(None, description="User's stated interest topic")
    userSummary: str | None = Field(default=None, description="Condensed summary of older messages")
    birthdate: Optional[str] = Field(None, description="User's birthdate in ISO format")
    topics: List[str] = Field(default_factory=list, description="List of user's interest topics")
