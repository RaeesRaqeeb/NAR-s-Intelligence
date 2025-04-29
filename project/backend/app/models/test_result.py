from datetime import datetime
from pydantic import BaseModel
from typing import List, Dict

class TestResultBase(BaseModel):
    user_id: str
    subject: str
    score: float
    total_questions: int
    correct_answers: int
    time_taken: int  # in seconds
    answers: Dict[str, str]  # question_id: selected_option_id

class TestResult(TestResultBase):
    id: str
    created_at: datetime = datetime.utcnow()

    class Config:
        from_attributes = True