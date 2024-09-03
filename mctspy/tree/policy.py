class AbstractPolicy:
    def get_action_proba_dist(self, state, action_mask) -> list[float]:
        return NotImplemented
