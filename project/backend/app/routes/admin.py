from fastapi import APIRouter, Depends, HTTPException
from ..models.user import User
from ..utils.security import get_current_user
from ..utils.database import db
from typing import List
from bson import ObjectId

router = APIRouter()

@router.get("/stats")
async def get_admin_stats(current_user: User = Depends(get_current_user)):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Forbidden")
    
    total_users = await db["users"].count_documents({})
    active_tests = await db["test_sessions"].count_documents({"completed": False})
    total_questions = await db["questions"].count_documents({})
    
    avg_score = await db["test_results"].aggregate([
        {"$group": {"_id": None, "avgScore": {"$avg": "$score"}}}
    ]).to_list(1)
    
    return {
        "total_users": total_users,
        "active_tests": active_tests,
        "total_questions": total_questions,
        "avg_score": round(avg_score[0]["avgScore"], 2) if avg_score else 0
    }

@router.get("/users")
async def get_users(
    page: int = 1,
    limit: int = 10,
    search: str = "",
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Forbidden")
    
    skip = (page - 1) * limit
    query = {}
    
    if search:
        query["$or"] = [
            {"email": {"$regex": search, "$options": "i"}},
            {"first_name": {"$regex": search, "$options": "i"}},
            {"last_name": {"$regex": search, "$options": "i"}}
        ]
    
    users = await db["users"].find(query).skip(skip).limit(limit).to_list(limit)
    total = await db["users"].count_documents(query)
    
    return {
        "users": users,
        "total": total,
        "page": page,
        "total_pages": (total + limit - 1) // limit
    }

@router.patch("/users/{user_id}/status")
async def toggle_user_status(
    user_id: str,
    is_active: bool,
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Forbidden")
    
    result = await db["users"].update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"is_active": is_active}}
    )
    
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {"message": "User status updated"}