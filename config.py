from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
import os


# 注意名字需要和配置的一样
class Settings(BaseSettings):
    
    if os.path.exists("config.env"):
        model_config = SettingsConfigDict(env_file="config.env")

    tcloud_secret_id: str = os.getenv("tcloud_secret_id","")
    tcloud_secret_key: str = os.getenv("tcloud_secret_key","")
    mysql_db_url: str = os.getenv("mysql_db_url", "")
    redis_host: str = os.getenv("redis_host", "localhost")
    redis_port: int = os.getenv("redis_port", 6379)
    redis_password: str = os.getenv("redis_password")
    is_tracemalloc: bool = os.getenv("is_tracemalloc", False)
    tcloud_appid: str = os.getenv("tcloud_appid", "")
    es_url: str = os.getenv("es_url", "")
    es_api_key: str = os.getenv("es_api_key","")
    lang_smith_api_key: str = os.getenv("lang_smith_api_key", "")
    anthropic_api_key: str = os.getenv("anthropic_api_key", "")


@lru_cache
def get_settings():
    return Settings()

# 会话相关常量
class ChatConstants:
    CHAT_ROLE_USER = "user"
    CHAT_ROLE_AI = "assistant"
