import math
import numpy as np


class OpponentBehaviorEstimator:
    def __init__(self, opp_max=1.0, opp_reservation=0.0):
        """
        Initializes the estimator.

        Parameters:
            opp_max (float): Maximum possible utility for the opponent.
            opp_reservation (float): Opponent's reservation value.
        """
        self.opp_max = opp_max
        self.opp_reservation = opp_reservation
        self.offers = []  # list of tuples: (t, observed_utility)
        self.alpha = None  # estimated concession parameter
        self.fit_quality = None  # residual sum of squares from regression
        self.my_offers = []  # record our offers for correlation analysis

    def add_offer(self, t, outcome, opp_ufun):
        """
        Records an opponent's offer and updates the concession parameter.

        Parameters:
            t (float): Normalized time of the offer (0 to 1).
            outcome: The offered outcome.
            opp_ufun (callable): Function to compute the opponent's utility for an outcome.
        """
        U = opp_ufun(outcome)
        self.offers.append((t, U))
        self._update_alpha()

    def add_my_offer(self, outcome):
        """
        Optionally record our own offer to later compute correlation with opponent's offers.
        """
        self.my_offers.append(outcome)

    def _update_alpha(self):
        """
        Updates the estimated concession parameter alpha using linear regression.
        Also computes a simple metric for the goodness of fit.

        The model:
            opp_max - U = (opp_max - opp_reservation) * t^alpha
        Taking logs:
            log(opp_max - U) = log(opp_max - opp_reservation) + alpha * log(t)
        """
        # Use only data with t > 0 and where U is less than opp_max.
        data = [(t, U) for (t, U) in self.offers if t > 0 and U < self.opp_max]
        if len(data) < 2:
            self.alpha = None
            self.fit_quality = None
            return

        xs, ys = [], []
        for t, U in data:
            denom = self.opp_max - self.opp_reservation + 1e-9
            xs.append(math.log(t))
            ys.append(math.log(self.opp_max - U) - math.log(denom))

        A = np.vstack([xs, np.ones(len(xs))]).T
        solution, residuals, _, _ = np.linalg.lstsq(A, ys, rcond=None)
        self.alpha = solution[0]
        self.fit_quality = residuals[0] if len(residuals) > 0 else None

    def get_concession_parameter(self):
        """Returns the current estimate of the concession parameter (alpha)."""
        return self.alpha

    def get_fit_quality(self):
        """Returns a measure of the goodness-of-fit (lower is better)."""
        return self.fit_quality

    def get_behavior_profile(self):
        """
        Returns a descriptive profile based on alpha and the fit quality.
        Also suggests if alternative behavior might be at play.
        """
        if self.alpha is None or self.fit_quality is None:
            return "Insufficient data for curve-fit analysis"
        # Threshold for a "good" fit; adjust based on domain specifics.
        if self.fit_quality > 0.5:
            return "Poor curve-fit: Opponent may not be following a smooth concession curve (e.g., Nash seeker or mirroring)"
        if self.alpha < 1:
            return "Boulware (tough, slow concession)"
        elif self.alpha > 1:
            return "Conceder (fast concession)"
        else:
            return "Linear concession"

    def analyze_mirroring(
        self, our_offer_history, opp_offer_history, opp_ufun
    ):
        """
        Analyzes if the opponent is mirroring our offers.
        It computes the correlation between our offers and opponent's offers
        (using their utilities) over time.

        Returns a correlation coefficient between -1 and 1.
        """
        if len(our_offer_history) < 2 or len(opp_offer_history) < 2:
            return None

        # Compute our offer utilities for the opponent.
        opp_util_my_offers = [opp_ufun(o) for o in our_offer_history]
        opp_util_opp_offers = [opp_ufun(o) for o in opp_offer_history]

        # Ensure both lists have the same length (e.g., pair by round order)
        n = min(len(opp_util_my_offers), len(opp_util_opp_offers))
        x = np.array(opp_util_my_offers[:n])
        y = np.array(opp_util_opp_offers[:n])
        if np.std(x) == 0 or np.std(y) == 0:
            return None
        correlation = np.corrcoef(x, y)[0, 1]
        return correlation


if __name__ == "__main__":
    # ---------------------
    # Example usage:
    # ---------------------

    # Assume the opponent's utility function is known:
    def opponent_ufun(outcome):
        # For illustration, assume outcome is a float directly representing utility.
        return outcome

    # Create an instance for a typical opponent with max utility 1.0 and reservation 0.2.
    estimator = OpponentBehaviorEstimator(opp_max=1.0, opp_reservation=0.2)

    # Simulate recording offers over time.
    estimator.add_offer(0.1, 0.95, opponent_ufun)
    estimator.add_offer(0.3, 0.90, opponent_ufun)
    estimator.add_offer(0.5, 0.82, opponent_ufun)
    estimator.add_offer(0.8, 0.70, opponent_ufun)

    alpha = estimator.get_concession_parameter()
    fit_quality = estimator.get_fit_quality()
    profile = estimator.get_behavior_profile()

    print("Estimated concession parameter (alpha):", alpha)
    print("Fit quality (residual):", fit_quality)
    print("Opponent behavior profile:", profile)

    # Optionally, analyze mirroring if you record your own offers.
    # Here we simulate our offers and opponent's offers over rounds.
    our_offers = [0.93, 0.92, 0.88, 0.85]
    opp_offers = [0.94, 0.91, 0.87, 0.84]
    correlation = estimator.analyze_mirroring(
        our_offers, opp_offers, opponent_ufun
    )
    print("Correlation between our offers and opponent's offers:", correlation)
