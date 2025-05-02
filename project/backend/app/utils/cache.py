from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
from redis import asyncio as aioredis
from typing import Optional, Callable, Any
import pickle
from datetime import timedelta
from ..config import settings

# Initialize cache at startup
async def init_cache():
    """
    Initialize Redis cache connection when application starts
    """
    try:
        redis = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=False
        )
        FastAPICache.init(
            RedisBackend(redis),
            prefix="academic-cache",
            key_builder=custom_key_builder
        )
        print("Redis cache initialized successfully")
    except Exception as e:
        print(f"Error initializing Redis cache: {e}")
        raise

def custom_key_builder(
    func: Callable,
    namespace: Optional[str] = "",
    *args,
    **kwargs
) -> str:
    """
    Custom cache key builder that includes:
    - Function module and name
    - Positional arguments
    - Keyword arguments
    - User role (if authenticated)
    """
    from fastapi import Request
    from fastapi.security import HTTPBearer
    
    # Extract request object if present
    request = None
    for arg in args:
        if isinstance(arg, Request):
            request = arg
            break
    
    # Get user role from token if available
    user_role = None
    if request:
        try:
            security = HTTPBearer()
            creds = security(request)
            if creds:
                token = creds.credentials
                payload = jwt.decode(
                    token,
                    settings.SECRET_KEY,
                    algorithms=[settings.ALGORITHM]
                )
                user_role = payload.get("role")
        except Exception:
            pass
    
    # Build the cache key
    key = (
        f"{func.__module__}:{func.__name__}:{namespace}:"
        f"{str(args)}:{str(kwargs)}"
    )
    if user_role:
        key += f":{user_role}"
    
    return key

def cache(
    expire: int = settings.CACHE_EXPIRE,
    namespace: str = "",
    key_builder: Callable = custom_key_builder
):
    """
    Enhanced cache decorator with:
    - Default expiration from settings
    - Custom key builder
    - Namespace support
    """
    return FastAPICache.cache(
        expire=expire,
        namespace=namespace,
        key_builder=key_builder
    )

async def clear_cache_pattern(pattern: str):
    """
    Clear cache entries matching a pattern
    """
    try:
        redis = FastAPICache.get_backend().redis
        keys = []
        async for key in redis.scan_iter(match=f"{FastAPICache.get_prefix()}:{pattern}:*"):
            keys.append(key)
        if keys:
            await redis.delete(*keys)
    except Exception as e:
        print(f"Error clearing cache pattern {pattern}: {e}")

async def get_cache(key: str) -> Any:
    """
    Directly get a cached value by key
    """
    try:
        redis = FastAPICache.get_backend().redis
        value = await redis.get(key)
        if value:
            return pickle.loads(value)
        return None
    except Exception as e:
        print(f"Error getting cache key {key}: {e}")
        return None

async def set_cache(
    key: str,
    value: Any,
    expire: Optional[int] = None
) -> bool:
    """
    Directly set a cached value with expiration
    """
    try:
        redis = FastAPICache.get_backend().redis
        if expire is None:
            expire = settings.CACHE_EXPIRE
        await redis.set(
            key,
            pickle.dumps(value),
            ex=timedelta(seconds=expire)
        )
        return True
    except Exception as e:
        print(f"Error setting cache key {key}: {e}")
        return False

async def invalidate_cache(*patterns: str):
    """
    Invalidate cache entries matching multiple patterns
    """
    for pattern in patterns:
        await clear_cache_pattern(pattern)