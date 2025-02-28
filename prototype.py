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


def aspiration_function(t, mx, rv, e):
    """Time-dependent aspiration function."""
    return (mx - rv) * (1.0 - np.power(t, e)) + rv


def average_step_time(state: SAOState) -> float:
    """Calculates the average step time to determine the last two steps."""
    return state.relative_time / max(1, state.step)


class AwesomeNegotiator(SAONegotiator):
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
        self.stochasticity = stochasticity
        self.min_unique_utilities = min_unique_utilities
        self.e = e
        self.nash_factor = nash_factor
        self.opp_offer_history = []
        self.my_offer_history = []
        self.opponent_utilities = []
        self.my_utilities = []
        self.mode = 0

    def on_preferences_changed(self, changes):
        if self.ufun is None:
            return
        self.rational_outcomes = [
            _ for _ in self.nmi.outcome_space.enumerate_or_sample()
            if self.ufun(_) > self.ufun.reserved_value
        ]
        self.partner_reserved_value = self.ufun.reserved_value

    def __call__(self, state: SAOState) -> SAOResponse:
        assert self.ufun and self.opponent_ufun

        if state.current_offer is not None:
            self.opp_offer_history.append(state.current_offer)

        if self.acceptance_strategy(state):
            return SAOResponse(ResponseType.ACCEPT_OFFER)
        return SAOResponse(ResponseType.REJECT_OFFER, self.bidding_strategy(state))

    def acceptance_strategy(self, state: SAOState) -> bool:
        offer = state.current_offer
        time_ratio = state.relative_time

        if self.ufun(offer) > (2 * self.ufun.reserved_value):
            return True
        if time_ratio >= self.time_dependent_threshold and self.ufun(offer) > self.ufun.reserved_value * 1.1:
            return True
        if time_ratio >= 0.99 and self.ufun(offer) > self.ufun.reserved_value * 1.1:
            return True
        return False

    def bidding_strategy(self, state: SAOState) -> Outcome | None:
        if len(state.history) < 2:
            return random.choice(self.rational_outcomes)
        
        last_offer = state.history[-1].offer
        prev_offer = state.history[-2].offer
        
        avg_step_time = average_step_time(state)
        if state.relative_time + avg_step_time >= 1.0:
            return self.best_opponent_offer(state)

        if self.opponent_is_time_dependent(last_offer, prev_offer):
            return self.predict_concession_price(state)
        return self.best_opponent_offer(state)

    def opponent_is_time_dependent(self, last_offer, prev_offer) -> bool:
        return last_offer is not None and prev_offer is not None and self.opponent_ufun(last_offer) < self.opponent_ufun(prev_offer)

    def predict_concession_price(self, state: SAOState) -> Outcome:
        if len(state.history) < 3:
            return random.choice(self.rational_outcomes)
        
        max_util = max(self.opponent_ufun(h.offer) for h in state.history[-3:])
        min_util = min(self.opponent_ufun(h.offer) for h in state.history[-3:])
        predicted = (max_util + min_util) / 2  # Linear interpolation approximation
        
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
