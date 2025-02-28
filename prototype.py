"""
**Submitted to ANAC 2024 Automated Negotiation League**
*Team* type your team name here
*Authors* type your team member names with their emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 ANL competition.
"""

import math
import random
import numpy as np
import negmas
from negmas.outcomes import Outcome
from negmas.sao import ResponseType, SAONegotiator, SAOResponse, SAOState
from opponent_behavior_estimator import OpponentBehaviorEstimator
import datetime


def aspiration_function(t, mx, rv, e):
    """Time-dependent aspiration function."""
    return (mx - rv) * (1.0 - np.power(t, e)) + rv


class AwesomeNegotiator(SAONegotiator):
    """
    A competitive negotiator that integrates opponent modeling,
    Pareto frontier/Nash analysis, and a time-based concession strategy.
    It dynamically switches its offering mode based on observed opponent behavior.
    """
    rational_outcomes = tuple()
    partner_reserved_value = 0
    time_dependent_threshold = 0.95

    def __init__(
        self,
        *args,
        stochasticity: float = 0.1,
        min_unique_utilities: int = 10,
        e: float = 17.5,
        nash_factor: float = 0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Stochasticity and concession parameters
        self.stochasticity = stochasticity
        self.min_unique_utilities = min_unique_utilities
        self.fe = e  # baseline concession exponent
        self.e = e  # current concession exponent (can be updated during negotiation)

        # Precompute outcome space once (assuming static domain)
        self._all_outcomes = (
            list(self.ufun.outcome_space.enumerate_or_sample())
            if self.ufun and self.ufun.outcome_space is not None
            else []
        )

        # Placeholders for Pareto frontier, Nash analysis, and rational outcomes
        self._pareto_outcomes = []
        self._nash_point = None
        self._rational = []

        # History for logging opponent and our offers (for opponent modeling)
        self.opp_offer_history = []
        self.my_offer_history = []
        self.opponent_utilities = []
        self.my_utilities = []

        # Partner reserved value (if needed in extensions)
        self.partner_reserved_value = 0.0
        self.mode = 0

    def on_preferences_changed(self, changes):
        """Called when preferences are updated. Computes outcome space, Pareto frontier,
        Nash point, and filters rational outcomes."""
        if self.ufun is None:
            return

        self.rational_outcomes = [
            _ for _ in self.nmi.outcome_space.enumerate_or_sample()
            if self.ufun(_) > self.ufun.reserved_value
        ]
        self.partner_reserved_value = self.ufun.reserved_value

    def on_negotiation_start(self, state):
        super().on_negotiation_start(state)

        # Initialize the opponent behavior estimator with logging enabled
        self._opp_estimator = OpponentBehaviorEstimator(
            opp_max=1.0,
            opp_reservation=self.partner_reserved_value,
            debug=True,
        )

    def __call__(self, state: SAOState) -> SAOResponse:
        """
        Main negotiation method.
        Updates opponent behavior, selects a negotiation mode,
        and chooses an offer accordingly.
        """
        assert self.ufun and self.opponent_ufun

        # Update opponent modeling with the current offer.
        if state.current_offer is not None:
            self._opp_estimator.add_offer(
                state.relative_time,
                state.current_offer,
                self.opponent_ufun,
            )
            self.opp_offer_history.append(state.current_offer)

        # Determine opponent behavior profile and mirroring correlation.
        # mirroring_corr = self._opp_estimator.analyze_mirroring(
        #     self.my_offer_history, self.opp_offer_history, self.opponent_ufun
        # )

        # Choose offer based on the selected mode.
        if self.mode == 0:
            # Mode 0: Use aspiration function and rational outcomes.
            asp = aspiration_function(
                state.relative_time, 1.0, self.ufun.reserved_value, self.e
            )
            offer = (
                self._rational[0][-1] if self._rational else self.ufun.best()
            )
        else:
            # Mode 1: Use Pareto/Nash-based offer selection.
            if self._nash_point is not None:
                offer = self._nash_point
            elif self._pareto_outcomes:
                offer = self._pareto_outcomes[-1]
            else:
                offer = self.ufun.best()

        # Record our offer for future modeling.
        self.my_offer_history.append(offer)
        self._opp_estimator.add_my_offer(state.relative_time, self.ufun(offer))
        self.my_utilities.append(float(self.ufun(offer)))

        self.nmi.log_info(self.id, self._opp_estimator.get_dynamic_profile())
        if self.acceptance_strategy(state):
            return SAOResponse(ResponseType.ACCEPT_OFFER)
        return SAOResponse(ResponseType.REJECT_OFFER, offer)

    def acceptance_strategy(self, state: SAOState) -> bool:
        """
        Accept the opponent's offer if it is sufficiently above our reservation.
        In this simple rule, if the offer's utility exceeds twice our reserved value, accept.
        """
        offer = state.current_offer
        time_ratio = state.relative_time

        if self.ufun(offer) > (2 * self.ufun.reserved_value):
            return True
        if time_ratio >= self.time_dependent_threshold and self.ufun(offer) > self.ufun.reserved_value * 1.1:
            return True
        return False
    
    def bidding_strategy(self, state: SAOState) -> Outcome | None:
        if len(state.history) < 2:
            return random.choice(self.rational_outcomes)
        
        last_offer = state.history[-1].offer
        prev_offer = state.history[-2].offer
        
        if self.opponent_is_time_dependent(last_offer, prev_offer):
            return self.predict_concession_price(state)
        return self.best_opponent_offer(state)

    def opponent_is_time_dependent(self, last_offer, prev_offer) -> bool:
        return last_offer is not None and prev_offer is not None and self.opponent_ufun(last_offer) < self.opponent_ufun(prev_offer)

    def predict_concession_price(self, state: SAOState) -> Outcome:
        if len(state.history) < 3:
            return random.choice(self.rational_outcomes)
        
        offers = [h.offer for h in state.history[-3:]]
        utilities = [self.opponent_ufun(o) for o in offers]
        
        from sklearn.linear_model import LinearRegression
        model = LinearRegression().fit(np.arange(len(utilities)).reshape(-1, 1), np.array(utilities).reshape(-1, 1))
        predicted = model.predict([[len(utilities)]])[0][0]
        
        return max(self.rational_outcomes, key=lambda o: self.opponent_ufun(o) if self.opponent_ufun(o) <= predicted else -1)

    def best_opponent_offer(self, state: SAOState) -> Outcome:
        return max(state.history, key=lambda h: self.opponent_ufun(h.offer)).offer

    def update_partner_reserved_value(self, state: SAOState) -> None:
        if self.opponent_ufun(state.current_offer) < self.partner_reserved_value:
            self.partner_reserved_value = float(self.opponent_ufun(state.current_offer)) / 2

        self.rational_outcomes = [
            _ for _ in self.rational_outcomes if self.opponent_ufun(_) > self.partner_reserved_value
        ]


# if you want to do a very small test, use the parameter small=True here. Otherwise, you can use the default parameters.
if __name__ == "__main__":
    from helpers.runner import run_a_tournament

    run_a_tournament(AwesomeNegotiator, small=True)