"""
Main application package initialization file.
Sets up the FastAPI application and imports all necessary components.
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .utils.database import connect_to_db, close_db_connection
from .utils.cache import init_cache

# Initialize FastAPI application
app = FastAPI(
    title="Academic Prep System API",
    description="API for the Academic Test Preparation System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import all routers
from .routes import auth, admin, user, test

# Include all API routers
app.include_router(auth.router)
app.include_router(admin.router)
app.include_router(user.router)
app.include_router(test.router)

@app.on_event("startup")
async def startup_event():
    """
    Initialize application services when starting up
    """
    await connect_to_db()
    await init_cache()
    # Add any other initialization here

@app.on_event("shutdown")
async def shutdown_event():
    """
    Clean up application services when shutting down
    """
    await close_db_connection()
    # Add any other cleanup here

# Optional: Health check endpoint
@app.get("/health", include_in_schema=False)
async def health_check():
    """
    Health check endpoint for monitoring
    """
    return {"status": "healthy"}

# Import all models to ensure they're registered
from .models import user, question, test_result  # noqa