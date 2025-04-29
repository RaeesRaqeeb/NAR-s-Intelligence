from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, validator
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from bson import ObjectId
from typing import Optional
import re

from ..utils.database import db
from ..config import settings
from ..models.user import User, UserCreate, UserRole

router = APIRouter(prefix="/auth", tags=["authentication"])

# Password context for hashing and verification
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# Token models
class Token(BaseModel):
    access_token: str
    token_type: str
    user_id: str
    role: str

class TokenData(BaseModel):
    id: Optional[str] = None
    role: Optional[str] = None

# Request models
class PasswordResetRequest(BaseModel):
    email: EmailStr

class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str

    @validator('new_password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not re.search("[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search("[0-9]", v):
            raise ValueError("Password must contain at least one number")
        if not re.search("[!@#$%^&*()_+]", v):
            raise ValueError("Password must contain at least one special character")
        return v

# Helper functions
def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

async def authenticate_user(email: str, password: str):
    user = await db["users"].find_one({"email": email})
    if not user:
        return False
    if not verify_password(password, user["password"]):
        return False
    return user

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        
        # Validate MongoDB ObjectId format
        if not ObjectId.is_valid(user_id):
            raise credentials_exception
            
        token_data = TokenData(id=user_id, role=payload.get("role"))
    except JWTError:
        raise credentials_exception
    
    user = await db["users"].find_one({"_id": ObjectId(token_data.id)})
    if user is None:
        raise credentials_exception
    
    # Convert MongoDB document to User model
    return User(**user)

# Endpoints
@router.post("/signup", response_model=User)
async def signup(user_data: UserCreate):
    """
    Register a new user account
    """
    # Check if user already exists
    existing_user = await db["users"].find_one({"email": user_data.email})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Validate password complexity
    password_errors = []
    if len(user_data.password) < 8:
        password_errors.append("Password must be at least 8 characters")
    if not re.search("[A-Z]", user_data.password):
        password_errors.append("Password must contain at least one uppercase letter")
    if not re.search("[0-9]", user_data.password):
        password_errors.append("Password must contain at least one number")
    if not re.search("[!@#$%^&*()_+]", user_data.password):
        password_errors.append("Password must contain at least one special character")
    
    if password_errors:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"password_errors": password_errors}
        )
    
    try:
        # Hash password and create user
        hashed_password = get_password_hash(user_data.password)
        user_dict = user_data.dict(exclude={"password"})
        user_dict.update({
            "password": hashed_password,
            "created_at": datetime.utcnow(),
            "is_active": True,
            "role": UserRole.USER.value  # Default role
        })

        result = await db["users"].insert_one(user_dict)
        user_dict["id"] = str(result.inserted_id)
        
        # Create a welcome event
        await db["user_events"].insert_one({
            "user_id": result.inserted_id,
            "type": "account_created",
            "timestamp": datetime.utcnow(),
            "data": {"email": user_data.email}
        })
        
        return user_dict
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating user: {str(e)}"
        )

@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Authenticate user and return access token
    """
    user = await authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive"
        )
    
    # Update last login time
    await db["users"].update_one(
        {"_id": user["_id"]},
        {"$set": {"last_login": datetime.utcnow()}}
    )
    
    # Create access token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user["_id"]), "role": user["role"]},
        expires_delta=access_token_expires
    )
    
    # Log login event
    await db["user_events"].insert_one({
        "user_id": user["_id"],
        "type": "login",
        "timestamp": datetime.utcnow(),
        "ip_address": None  # Would be populated from request in production
    })
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": str(user["_id"]),
        "role": user["role"]
    }

@router.post("/password/reset/request")
async def request_password_reset(request: PasswordResetRequest):
    """
    Request a password reset (sends email with reset token)
    """
    user = await db["users"].find_one({"email": request.email})
    if not user:
        # Don't reveal whether email exists for security
        return {"message": "If the email exists, a reset link has been sent"}
    
    # Create reset token (expires in 1 hour)
    reset_token = create_access_token(
        data={"sub": str(user["_id"]), "purpose": "password_reset"},
        expires_delta=timedelta(hours=1)
    )
    
    # In production, you would send an email here
    # For this example, we'll just return the token
    # NEVER do this in production!
    return {
        "message": "Password reset token generated",
        "reset_token": reset_token  # Remove this in production!
    }

@router.post("/password/reset/confirm")
async def confirm_password_reset(confirm: PasswordResetConfirm):
    """
    Confirm password reset with token and new password
    """
    try:
        payload = jwt.decode(
            confirm.token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        if payload.get("purpose") != "password_reset":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid token purpose"
            )
        
        user_id = payload.get("sub")
        if not user_id or not ObjectId.is_valid(user_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid user in token"
            )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired token"
        )
    
    # Update password
    hashed_password = get_password_hash(confirm.new_password)
    result = await db["users"].update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"password": hashed_password}}
    )
    
    if result.modified_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Log password change event
    await db["user_events"].insert_one({
        "user_id": ObjectId(user_id),
        "type": "password_reset",
        "timestamp": datetime.utcnow()
    })
    
    return {"message": "Password updated successfully"}

@router.get("/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    """
    Get current user information
    """
    return current_user

@router.post("/logout")
async def logout(current_user: User = Depends(get_current_user)):
    """
    Log out current user (client should discard token)
    """
    # In a real implementation, you might add the token to a blacklist
    # Here we just log the event
    await db["user_events"].insert_one({
        "user_id": ObjectId(current_user.id),
        "type": "logout",
        "timestamp": datetime.utcnow()
    })
    
    return {"message": "Successfully logged out"}