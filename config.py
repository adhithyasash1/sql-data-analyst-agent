import hashlib
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def default_metadata_path_for_source(source_db_path: Path) -> Path:
    """Derive a deterministic, per-source-database metadata DB path.

    The path is keyed on the source database's absolute path so that distinct
    databases (even same-named ones in different directories) never share a
    metadata/index file. The filename stays path-only so query logs remain
    stable across embedding-model or schema changes; model/dimension/fingerprint
    checks live inside the index state and are handled by ensure_index_ready().
    """
    resolved = source_db_path.expanduser().resolve()
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:12]
    safe_stem = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in resolved.stem
    )
    return Path("data/metadata") / f"{safe_stem}-{digest}.metadata.db"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    omlx_base_url: str = "http://127.0.0.1:8000/v1"
    omlx_api_key: str = Field(default="")
    text_to_sql_model: str = "MLX-Qwopus3.5-9B-v3-4bit"
    summary_model: str = "MLX-Qwopus3.5-9B-v3-4bit"
    embedding_model: str = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
    db_path: Path = Path("data/northwind.db")
    metadata_db_path: Path | None = None
    retrieval_top_k: int = 6
    max_result_rows: int = 50
    max_sql_tokens: int = 2048
    max_summary_tokens: int = 512
    sql_temperature: float = 0.0
    sql_top_p: float = 1.0
    summary_temperature: float = 0.2
    summary_top_p: float = 0.9
    embedding_batch_size: int = 16
    query_timeout_ms: int = 3000
    max_repair_attempts: int = 1
    auto_reindex: bool = True
    enable_llm_summary: bool = False
    enable_result_shape_check: bool = True
    enable_query_logging: bool = True
    require_sql_approval: bool = False

    @field_validator(
        "retrieval_top_k",
        "max_result_rows",
        "max_sql_tokens",
        "max_summary_tokens",
        "embedding_batch_size",
        "query_timeout_ms",
    )
    @classmethod
    def must_be_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be greater than 0")
        return value

    @field_validator("max_repair_attempts")
    @classmethod
    def repair_attempts_are_bounded(cls, value: int) -> int:
        if value < 0 or value > 3:
            raise ValueError("must be between 0 and 3")
        return value

    @field_validator("sql_temperature", "summary_temperature")
    @classmethod
    def temperature_range(cls, value: float) -> float:
        if value < 0 or value > 2:
            raise ValueError("must be between 0 and 2")
        return value

    @field_validator("sql_top_p", "summary_top_p")
    @classmethod
    def top_p_range(cls, value: float) -> float:
        if value <= 0 or value > 1:
            raise ValueError("must be greater than 0 and at most 1")
        return value

    @property
    def source_db_path(self) -> Path:
        return self.db_path.expanduser().resolve()

    @property
    def metadata_path(self) -> Path:
        if self.metadata_db_path is not None:
            return self.metadata_db_path.expanduser().resolve()
        return default_metadata_path_for_source(self.source_db_path).resolve()

    def validate_runtime(self) -> None:
        if not self.omlx_api_key or self.omlx_api_key == "replace-me":
            raise ValueError("Set OMLX_API_KEY in .env to any non-empty local API key value.")
