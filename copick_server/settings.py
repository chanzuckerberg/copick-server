from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    @classmethod
    def load(cls):
        return cls()

    PORT: int = 8000
    HOST: str = "127.0.0.1"
    CORS: str = "*"
    DEBUG: bool = True
    DATASET_IDS: list[str] = [] #["10440", "10445", "10446"]
    OVERLAY_ROOT: str = "/tmp/overlay_root"
    CONFIG: str = "example_copick.json"
