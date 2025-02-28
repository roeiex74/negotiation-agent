import math
import numpy as np
from collections import defaultdict


class OpponentBehaviorEstimator:
    def __init__(self, opp_max=1.0, opp_reservation=0.0, debug=False):
        """
        A dynamic opponent behavior estimator for ANL negotiations.
        Tracks key negotiation metrics without forcing pre-classification.
        """
        self.opp_max = opp_max
        self.opp_reservation = opp_reservation
        self.offers = []  # List of tuples (t, observed_utility)
        self.my_offers = []  # List of my own offers (t, utility)
        self.alpha = None  # Concession rate
        self.fit_quality = None  # Residual sum of squares from regression
        self.alpha_history = []  # Tracks alpha trends
        self.mirroring_correlation = None  # Correlation with opponent's offers
        self.diff_concession = 0  # My total concession
        self.opp_diff_concession = 0  # Opponent's total concession
        self.debug = False

    def add_offer(self, t, outcome, opp_ufun):
        """Records an opponent's offer and updates concession calculations."""
        U = opp_ufun(outcome)
        self.offers.append((t, U))
        self._update_alpha()

    def add_my_offer(self, t, my_utility):
        """Records my own offer for later analysis."""
        self.my_offers.append((t, my_utility))

    def _update_alpha(self):
        """Updates the estimated concession parameter alpha using linear regression."""
        data = [(t, U) for (t, U) in self.offers if t > 0 and U < self.opp_max]
        if len(data) < 2:
            self.alpha = None
            self.fit_quality = None
            return
        xs, ys = zip(
            *[(math.log(t), math.log(self.opp_max - U)) for t, U in data]
        )
        A = np.vstack([xs, np.ones(len(xs))]).T
        solution, residuals, _, _ = np.linalg.lstsq(A, ys, rcond=None)
        self.alpha = solution[0]
        self.alpha_history.append(self.alpha)
        self.fit_quality = residuals[0] if len(residuals) > 0 else None

    def analyze_mirroring(self):
        """Analyzes if the opponent is mirroring our offers."""
        if len(self.my_offers) < 2 or len(self.offers) < 2:
            return None
        my_utils = [u for _, u in self.my_offers]
        opp_utils = [u for _, u in self.offers]
        min_len = min(len(my_utils), len(opp_utils))
        x, y = np.array(my_utils[:min_len]), np.array(opp_utils[:min_len])
        if np.std(x) == 0 or np.std(y) == 0:
            return None
        self.mirroring_correlation = np.corrcoef(x, y)[0, 1]
        return self.mirroring_correlation

    def compute_concession_friendly(self):
        """Calculates whether the opponent is likely to be concession-friendly."""
        if len(self.offers) < 2 or len(self.my_offers) < 2:
            return False  # Not enough data

        # Compute initial and final utility values
        my_start, my_end = self.my_offers[0][1], self.my_offers[-1][1]
        opp_start, opp_end = self.offers[0][1], self.offers[-1][1]

        self.diff_concession = my_start - my_end  # My concession
        self.opp_diff_concession = opp_start - opp_end  # Opponent's concession

        # threshold check for a possible concessions based opponent
        return (self.diff_concession - self.opp_diff_concession) > 0.2

    def get_dynamic_profile(self):
        """Provides real-time metrics instead of static classification."""
        return {
            "alpha": self.alpha,
            # "alpha_trend": self.alpha_history,
            "fit_quality": self.fit_quality,
            "mirroring": self.mirroring_correlation,
            "our_diff": self.diff_concession,
            "opp_diff": self.opp_diff_concession,
            "concession_friendly": self.compute_concession_friendly(),
            "my_latest_offer": self.my_offers[-1][1],
            "opp_latest_offer": self.offers[-1][1],
        }
