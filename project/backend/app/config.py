from pydantic import BaseSettings

class Settings(BaseSettings):
    SECRET_KEY: str = "your-secret-key-here"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    MONGODB_URI: str = "mongodb://localhost:27017/academic_prep"
    CORS_ORIGINS: list = ["*"]

    class Config:
        env_file = ".env"

settings = Settings()