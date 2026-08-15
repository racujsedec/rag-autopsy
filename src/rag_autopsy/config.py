from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int = 120
    overlap: int = 20

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if self.overlap < 0:
            raise ValueError("overlap cannot be negative")
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
