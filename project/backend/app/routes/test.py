from fastapi import APIRouter, Depends, HTTPException
from ..models.test_result import TestResult, TestResultBase
from ..models.question import Question, DifficultyLevel, Subject
from ..utils.database import db
from ..models.user import User
from ..utils.security import get_current_user
from typing import List
import random

router = APIRouter()

@router.get("/questions", response_model=List[Question])
async def get_questions(
    subject: Subject,
    difficulty: DifficultyLevel,
    limit: int = 10,
    current_user: User = Depends(get_current_user)
):
    questions = await db["questions"].find(
        {"subject": subject, "difficulty": difficulty}
    ).to_list(limit)
    random.shuffle(questions)
    return questions

@router.post("/submit", response_model=TestResult)
async def submit_test(
    test_data: TestResultBase,
    current_user: User = Depends(get_current_user)
):
    test_dict = test_data.dict()
    test_dict["user_id"] = str(current_user.id)
    
    result = await db["test_results"].insert_one(test_dict)
    test_dict["id"] = str(result.inserted_id)
    
    return test_dict

@router.get("/results", response_model=List[TestResult])
async def get_user_results(
    current_user: User = Depends(get_current_user),
    limit: int = 10,
    skip: int = 0
):
    results = await db["test_results"].find(
        {"user_id": str(current_user.id)}
    ).skip(skip).limit(limit).to_list()
    return results