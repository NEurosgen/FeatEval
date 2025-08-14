# selectors/search.py
class SubsetSearch(Protocol):
    def run(self, 
            X, y, base_cols: list[str], pool_cols: list[str],
            cv: CVEngine, scorer, beam_width: int, early_stop_patience: int
    ) -> list[str]