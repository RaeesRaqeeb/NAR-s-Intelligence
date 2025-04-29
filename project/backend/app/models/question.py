from enum import Enum
from pydantic import BaseModel
from typing import List

class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class Subject(str, Enum):
    MATH = "math"
    SCIENCE = "science"
    ENGLISH = "english"
    HISTORY = "history"

class QuestionOption(BaseModel):
    text: str
    is_correct: bool = False

class QuestionBase(BaseModel):
    text: str
    subject: Subject
    difficulty: DifficultyLevel
    options: List[QuestionOption]
    explanation: str

class Question(QuestionBase):
    id: str
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True