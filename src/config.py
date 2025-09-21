from dataclasses import dataclass


@dataclass
class Config:
    top_percentile: float = 80.0
    max_fetch_count: int = 10


if __name__ == "__main__":
    print(Config.top_percentile)
