class Ranker(Protocol):
    def rank(self, X, y, candidates: list[str], conditioned_on: list[str] | None) -> list[str]
