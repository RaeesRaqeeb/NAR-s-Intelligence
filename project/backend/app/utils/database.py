from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.encoders import jsonable_encoder
from typing import List, Optional
from bson import ObjectId
from datetime import datetime
from pydantic import BaseModel, EmailStr, validator
from ..models.user import User, UserCreate, UserRole
from ..models.question import Question, QuestionBase, QuestionOption, Subject, DifficultyLevel
from ..utils.security import get_current_user, get_current_admin
from ..utils.database import db
from ..utils.cache import cache
from ..config import settings

router = APIRouter(prefix="/admin", tags=["admin"])

# Response Models
class UserStats(BaseModel):
    total_users: int
    active_users: int
    new_users_today: int

class TestStats(BaseModel):
    total_tests: int
    avg_score: float
    tests_today: int

class AdminDashboardStats(BaseModel):
    users: UserStats
    tests: TestStats
    questions_by_subject: dict
    recent_activity: list

# Endpoints
@router.get("/dashboard", response_model=AdminDashboardStats)
@cache(expire=60)  # Cache for 1 minute
async def get_admin_dashboard(current_user: User = Depends(get_current_admin)):
    """
    Get comprehensive admin dashboard statistics
    """
    try:
        # User Statistics
        total_users = await db["users"].count_documents({})
        active_users = await db["users"].count_documents({"is_active": True})
        new_users_today = await db["users"].count_documents({
            "created_at": {"$gte": datetime.utcnow().replace(hour=0, minute=0, second=0)}
        })

        # Test Statistics
        total_tests = await db["test_results"].count_documents({})
        avg_score_result = await db["test_results"].aggregate([
            {"$group": {"_id": None, "avgScore": {"$avg": "$score"}}}
        ]).to_list(1)
        avg_score = round(avg_score_result[0]["avgScore"], 2) if avg_score_result else 0
        tests_today = await db["test_results"].count_documents({
            "created_at": {"$gte": datetime.utcnow().replace(hour=0, minute=0, second=0)}
        })

        # Questions by Subject
        questions_by_subject = await db["questions"].aggregate([
            {"$group": {"_id": "$subject", "count": {"$sum": 1}}}
        ]).to_list(None)

        # Recent Activity (last 10 actions)
        recent_activity = await db["admin_logs"].find().sort("timestamp", -1).limit(10).to_list(10)

        return {
            "users": {
                "total_users": total_users,
                "active_users": active_users,
                "new_users_today": new_users_today
            },
            "tests": {
                "total_tests": total_tests,
                "avg_score": avg_score,
                "tests_today": tests_today
            },
            "questions_by_subject": {item["_id"]: item["count"] for item in questions_by_subject},
            "recent_activity": recent_activity
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching dashboard stats: {str(e)}"
        )

@router.get("/users", response_model=List[User])
async def list_users(
    page: int = Query(1, ge=1),
    limit: int = Query(10, le=100),
    search: Optional[str] = None,
    role: Optional[UserRole] = None,
    is_active: Optional[bool] = None,
    current_user: User = Depends(get_current_admin)
):
    """
    List users with pagination and filtering
    """
    query = {}
    if search:
        query["$or"] = [
            {"email": {"$regex": search, "$options": "i"}},
            {"first_name": {"$regex": search, "$options": "i"}},
            {"last_name": {"$regex": search, "$options": "i"}}
        ]
    if role:
        query["role"] = role
    if is_active is not None:
        query["is_active"] = is_active

    try:
        users = await db["users"].find(
            query,
            {"password": 0}  # Exclude password hash
        ).skip((page - 1) * limit).limit(limit).to_list(limit)
        
        return users
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching users: {str(e)}"
        )

@router.get("/users/{user_id}", response_model=User)
async def get_user(
    user_id: str,
    current_user: User = Depends(get_current_admin)
):
    """
    Get detailed information about a specific user
    """
    if not ObjectId.is_valid(user_id):
        raise HTTPException(status_code=400, detail="Invalid user ID format")

    try:
        user = await db["users"].find_one(
            {"_id": ObjectId(user_id)},
            {"password": 0}
        )
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return user
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching user: {str(e)}"
        )

@router.post("/users", response_model=User)
async def create_user(
    user_data: UserCreate,
    current_user: User = Depends(get_current_admin)
):
    """
    Create a new user (admin only)
    """
    # Check if user already exists
    existing_user = await db["users"].find_one({"email": user_data.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    try:
        # Hash password and create user
        hashed_password = get_password_hash(user_data.password)
        user_dict = user_data.dict(exclude={"password"})
        user_dict.update({
            "password": hashed_password,
            "created_at": datetime.utcnow(),
            "is_active": True
        })

        result = await db["users"].insert_one(user_dict)
        user_dict["id"] = str(result.inserted_id)
        
        # Log admin action
        await log_admin_action(
            current_user.id,
            "create_user",
            f"Created user {user_dict['email']}"
        )
        
        return user_dict
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating user: {str(e)}"
        )

@router.patch("/users/{user_id}/status")
async def update_user_status(
    user_id: str,
    is_active: bool,
    current_user: User = Depends(get_current_admin)
):
    """
    Activate or deactivate a user account
    """
    if not ObjectId.is_valid(user_id):
        raise HTTPException(status_code=400, detail="Invalid user ID format")

    try:
        result = await db["users"].update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"is_active": is_active}}
        )
        
        if result.modified_count == 0:
            raise HTTPException(status_code=404, detail="User not found or no changes made")
        
        action = "activated" if is_active else "deactivated"
        await log_admin_action(
            current_user.id,
            "update_user_status",
            f"{action} user {user_id}"
        )
        
        return {"message": f"User {action} successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating user status: {str(e)}"
        )

@router.get("/questions", response_model=List[Question])
async def list_questions(
    page: int = Query(1, ge=1),
    limit: int = Query(10, le=100),
    subject: Optional[Subject] = None,
    difficulty: Optional[DifficultyLevel] = None,
    current_user: User = Depends(get_current_admin)
):
    """
    List questions with filtering and pagination
    """
    query = {}
    if subject:
        query["subject"] = subject
    if difficulty:
        query["difficulty"] = difficulty

    try:
        questions = await db["questions"].find(query).skip((page - 1) * limit).limit(limit).to_list(limit)
        return questions
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching questions: {str(e)}"
        )

@router.post("/questions", response_model=Question)
async def create_question(
    question: QuestionBase,
    current_user: User = Depends(get_current_admin)
):
    """
    Create a new test question
    """
    # Validate at least one correct option
    if not any(opt.is_correct for opt in question.options):
        raise HTTPException(status_code=400, detail="At least one option must be correct")

    try:
        question_dict = question.dict()
        question_dict.update({
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "created_by": str(current_user.id)
        })

        result = await db["questions"].insert_one(question_dict)
        question_dict["id"] = str(result.inserted_id)
        
        await log_admin_action(
            current_user.id,
            "create_question",
            f"Created question in {question.subject}/{question.difficulty}"
        )
        
        return question_dict
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating question: {str(e)}"
        )

@router.delete("/questions/{question_id}")
async def delete_question(
    question_id: str,
    current_user: User = Depends(get_current_admin)
):
    """
    Delete a test question
    """
    if not ObjectId.is_valid(question_id):
        raise HTTPException(status_code=400, detail="Invalid question ID format")

    try:
        result = await db["questions"].delete_one({"_id": ObjectId(question_id)})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Question not found")
        
        await log_admin_action(
            current_user.id,
            "delete_question",
            f"Deleted question {question_id}"
        )
        
        return {"message": "Question deleted successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting question: {str(e)}"
        )

@router.get("/test-results", response_model=List[TestResult])
async def list_test_results(
    page: int = Query(1, ge=1),
    limit: int = Query(10, le=100),
    user_id: Optional[str] = None,
    subject: Optional[Subject] = None,
    min_score: Optional[float] = None,
    max_score: Optional[float] = None,
    current_user: User = Depends(get_current_admin)
):
    """
    List test results with advanced filtering
    """
    query = {}
    if user_id:
        if not ObjectId.is_valid(user_id):
            raise HTTPException(status_code=400, detail="Invalid user ID format")
        query["user_id"] = user_id
    if subject:
        query["subject"] = subject
    if min_score is not None:
        query["score"] = {"$gte": min_score}
    if max_score is not None:
        query["score"] = {"$lte": max_score}

    try:
        results = await db["test_results"].find(query).skip((page - 1) * limit).limit(limit).to_list(limit)
        return results
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching test results: {str(e)}"
        )

# Helper Functions
async def log_admin_action(admin_id: str, action_type: str, description: str):
    """
    Log admin actions for audit trail
    """
    await db["admin_logs"].insert_one({
        "admin_id": admin_id,
        "action_type": action_type,
        "description": description,
        "timestamp": datetime.utcnow()
    })

def get_password_hash(password: str):
    return pwd_context.hash(password)

# Dependency for admin-only access
async def get_current_admin(current_user: User = Depends(get_current_user)):
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user