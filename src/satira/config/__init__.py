from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SATIRA_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Model config
    d_model: int = 512
    num_heads: int = 8
    num_classes: int = 5
    vision_dim: int = 1024
    text_dim: int = 768
    temporal_dim: int = 768
    graph_dim: int = 256
    dropout_rate: float = 0.2
    num_reasoning_layers: int = 2

    # Training config
    batch_size: int = 64
    learning_rate: float = 1e-3
    focal_gamma: float = 2.0
    gate_loss_weight: float = 0.3
    consistency_loss_weight: float = 0.1
    temporal_drop_prob: float = 0.15
    graph_drop_prob: float = 0.15
    joint_drop_prob: float = 0.05

    # Infrastructure config
    redis_url: str = "redis://localhost:6379"
    database_url: str = "postgresql://localhost:5432/satira"
    s3_bucket: str = "satira-data"
    faiss_index_path: str = "./data/faiss_index"
    graph_embeddings_path: str = "./data/graph_embeddings"
    max_batch_size: int = 32
    batch_timeout_ms: float = 50.0

    # Class gate targets
    CLASS_GATE_TARGETS: dict = {0: 0.1, 1: 0.9, 2: 0.8, 3: 0.2, 4: 0.4}
    CLASS_NAMES: list = [
        "authentic",
        "satire",
        "parody",
        "misleading_context",
        "fabricated",
    ]


settings = Settings()
