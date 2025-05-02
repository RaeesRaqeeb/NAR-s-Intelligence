import os
from datetime import datetime, timedelta
from typing import Optional, Union
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends, Request
from fastapi.security import OAuth2PasswordBearer, HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from bson import ObjectId
import secrets
import string
from ..config import settings
from ..models.user import User
import logging
from email_validator import validate_email, EmailNotValidError

# Configure logging
logger = logging.getLogger(__name__)

# Security constants
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES
REFRESH_TOKEN_EXPIRE_DAYS = 7
PASSWORD_RESET_TOKEN_EXPIRE_HOURS = 1

# Password context
pwd_context = CryptContext(
    schemes=["bcrypt", "argon2"],
    deprecated="auto",
    argon2__time_cost=3,
    argon2__memory_cost=65536,
    argon2__parallelism=4,
    argon2__hash_len=32
)

# OAuth2 schemes
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")
bearer_scheme = HTTPBearer()

class TokenPayload(BaseModel):
    sub: str  # user ID
    role: str
    exp: int
    iat: int
    jti: str  # unique token identifier

class TokenData(BaseModel):
    id: Optional[str] = None
    role: Optional[str] = None

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain password against a hashed password
    """
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        logger.error(f"Password verification failed: {str(e)}")
        return False

def get_password_hash(password: str) -> str:
    """
    Generate a secure hash from a plain password
    """
    return pwd_context.hash(password)

def generate_secure_token(length: int = 32) -> str:
    """
    Generate a cryptographically secure random token
    """
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None,
    token_type: str = "access"
) -> str:
    """
    Create a signed JWT token with expiration
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "jti": generate_secure_token(),
        "type": token_type
    })
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=ALGORITHM
    )
    return encoded_jwt

def decode_token(token: str) -> TokenPayload:
    """
    Decode and validate a JWT token
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[ALGORITHM]
        )
        token_data = TokenPayload(**payload)
        
        # Additional validation
        if not ObjectId.is_valid(token_data.sub):
            raise credentials_exception
            
        if token_data.exp < datetime.utcnow().timestamp():
            raise credentials_exception
            
        return token_data
    except (JWTError, AttributeError) as e:
        logger.error(f"Token validation failed: {str(e)}")
        raise credentials_exception

async def get_current_user(
    token: str = Depends(oauth2_scheme)
) -> User:
    """
    Dependency to get current authenticated user from JWT token
    """
    token_data = decode_token(token)
    
    try:
        user = await db["users"].find_one({"_id": ObjectId(token_data.sub)})
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
            
        return User(**user)
    except Exception as e:
        logger.error(f"Error fetching user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching user data"
        )

async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Dependency to verify user is active
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    return current_user

async def get_current_admin(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """
    Dependency to verify user has admin role
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user

def validate_email_address(email: str) -> bool:
    """
    Validate an email address using DNS verification
    """
    try:
        v = validate_email(email, check_deliverability=True)
        return v.email == email
    except EmailNotValidError as e:
        logger.warning(f"Email validation failed: {str(e)}")
        return False

def generate_csrf_token() -> str:
    """
    Generate a CSRF token for form protection
    """
    return secrets.token_urlsafe(32)

def verify_csrf_token(token: str, request: Request) -> bool:
    """
    Verify a CSRF token against the session
    """
    session_token = request.session.get("csrf_token")
    if not session_token:
        return False
    return secrets.compare_digest(token, session_token)

def get_authorization_header(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
) -> str:
    """
    Extract and validate Authorization header
    """
    if credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication scheme"
        )
    return credentials.credentials

def create_password_reset_token(user_id: str) -> str:
    """
    Create a time-limited password reset token
    """
    return create_access_token(
        {"sub": user_id, "purpose": "password_reset"},
        expires_delta=timedelta(hours=PASSWORD_RESET_TOKEN_EXPIRE_HOURS)
    )

def verify_password_reset_token(token: str) -> Optional[str]:
    """
    Verify and extract user ID from password reset token
    """
    try:
        payload = decode_token(token)
        if payload.get("purpose") != "password_reset":
            return None
        return payload.sub
    except Exception:
        return None

def validate_password_complexity(password: str) -> dict:
    """
    Validate password meets complexity requirements
    Returns dict with 'valid' bool and 'messages' list
    """
    errors = []
    
    if len(password) < 8:
        errors.append("Password must be at least 8 characters")
    if not any(c.isupper() for c in password):
        errors.append("Password must contain at least one uppercase letter")
    if not any(c.isdigit() for c in password):
        errors.append("Password must contain at least one digit")
    if not any(c in string.punctuation for c in password):
        errors.append("Password must contain at least one special character")
    
    return {
        "valid": len(errors) == 0,
        "messages": errors
    }

# Initialize database connection
from ..utils.database import db