"""
**Submitted to ANAC 2024 Automated Negotiation League**
*Team* type your team name here
*Authors* type your team member names with their emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 ANL competition.
"""

import numpy as np
import random
import negmas
from negmas.outcomes import Outcome
from negmas.sao import ResponseType, SAONegotiator, SAOResponse, SAOState
from copy import deepcopy
from opponent_behavior_estimator import OpponentBehaviorEstimator


def aspiration_function(t, mx, rv, e):
    return (mx - rv) * (1.0 - np.power(t, e)) + rv


class AwesomeNegotiator(SAONegotiator):
    """
    Your agent code. This is the ONLY class you need to implement
    """

    def __init__(
        self,
        *args,
        stochasticity: float = 0.1,
        min_unique_utilities: int = 10,
        e: float = 17.5,
        nash_factor: float = 0.1,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        # Stochasticity parameter
        self.stochasticity = stochasticity
        # Parameters for learning opponent behavior
        self.min_unique_utilities = min_unique_utilities
        self.fe = e  # baseline concession exponent
        self.e = e  # current exponent, may be updated during negotiation

        # Outcome-related initialization (compute outcome space once)
        self._all_outcomes = (
            list(self.ufun.outcome_space.enumerate_or_sample())
            if self.ufun and self.ufun.outcome_space is not None
            else []
        )

        # Placeholders for Pareto frontier and Nash analysis (will be computed later)
        self._pareto_outcomes = []
        self._nash_point = None

        # For rational outcome filtering and mode switching
        self._rational = (
            []
        )  # to store outcomes that are acceptable for both sides
        self.mode = 0  # a variable to switch modes based on observed negotiation dynamics

        # Logging data for analysis (if needed)
        self.opponent_utilities = []
        self.my_utilities = []

        self.partner_reserved_value = 0.0

        # Initialize opponent behavior estimator
        self._opp_estimator = OpponentBehaviorEstimator(
            opp_max=1.0, opp_reservation=self.partner_reserved_value
        )

    # rational_outcomes = tuple()

    # partner_reserved_value = 0

    def on_preferences_changed(self, changes):
        """
        Called when preferences change. In ANL 2024, this is equivalent with initializing the agent.

        Remarks:
            - Can optionally be used for initializing your agent.
            - We use it to save a list of all rational outcomes.

        """
        # If there a no outcomes (should in theory never happen)
        if self.ufun is None:
            return

        self.rational_outcomes = [
            _
            for _ in self.nmi.outcome_space.enumerate_or_sample()  # enumerates outcome space when finite, samples when infinite
            if self.ufun(_) > self.ufun.reserved_value
        ]

        # Estimate the reservation value, as a first guess, the opponent has the same reserved_value as you
        self.partner_reserved_value = self.ufun.reserved_value

    def __call__(self, state: SAOState) -> SAOResponse:
        """
        Called to (counter-)offer.

        Args:
            state: the `SAOState` containing the offer from your partner (None if you are just starting the negotiation)
                   and other information about the negotiation (e.g. current step, relative time, etc).
        Returns:
            A response of type `SAOResponse` which indicates whether you accept, or reject the offer or leave the negotiation.
            If you reject an offer, you are required to pass a counter offer.

        Remarks:
            - This is the ONLY function you need to implement.
            - You can access your ufun using `self.ufun`.
            - You can access the opponent's ufun using self.opponent_ufun(offer)
            - You can access the mechanism for helpful functions like sampling from the outcome space using `self.nmi` (returns an `SAONMI` instance).
            - You can access the current offer (from your partner) as `state.current_offer`.
              - If this is `None`, you are starting the negotiation now (no offers yet).
        """
        offer = state.current_offer

        self.update_partner_reserved_value(state)

        # if there are no outcomes (should in theory never happen)
        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        # my extention -is the proposal good enough?
        time_factor = 1 - state.relative_time  # Time based concessions
        acceptance_threshold = (
            self.ufun.reserved_value
            + (self.ufun.max() - self.ufun.reserved_value) * time_factor
        )

        if offer is not None and self.ufun(offer) >= acceptance_threshold:
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        # another extantion- counter offerf - finging the most optimal offer the other side might get
        possible_outcomes = self.nmi.outcome_space.enumerate_or_sample()
        # sort the offers according to ower value in decreasing order
        possible_outcomes = sorted(
            possible_outcomes, key=self.ufun, reverse=True
        )
        # select the best counter offer
        best_offer = None
        for outcome in possible_outcomes:
            if self.opponent_ufun(outcome) >= self.partner_reserved_value:
                best_offer = outcome
                break
        # if there is not best offer we will choose the closest one to the minimal val of the copponent
        if best_offer is None:
            best_offer = min(
                possible_outcomes,
                key=lambda x: abs(
                    self.opponent_ufun(x) - self.partner_reserved_value
                ),
            )
        # Determine the acceptability of the offer in the acceptance_strategy
        if self.acceptance_strategy(state):
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        # If it's not acceptable, determine the counter offer in the bidding_strategy
        return SAOResponse(
            ResponseType.REJECT_OFFER, self.bidding_strategy(state)
        )

    def acceptance_strategy(self, state: SAOState) -> bool:
        """
        This is one of the functions you need to implement.
        It should determine whether or not to accept the offer.

        Returns: a bool.
        """
        assert self.ufun

        offer = state.current_offer

        if self.ufun(offer) > (2 * self.ufun.reserved_value):
            return True
        return False

    def bidding_strategy(self, state: SAOState) -> Outcome | None:
        """
        This is one of the functions you need to implement.
        It should determine the counter offer.

        Returns: The counter offer as Outcome.
        """

        # The opponent's ufun can be accessed using self.opponent_ufun, which is not used yet.

        return random.choice(self.rational_outcomes)

    def update_partner_reserved_value(self, state: SAOState) -> None:
        """This is one of the functions you can implement.
        Using the information of the new offers, you can update the estimated reservation value of the opponent.

        returns: None.
        """
        assert self.ufun and self.opponent_ufun

        offer = state.current_offer

        if self.opponent_ufun(offer) < self.partner_reserved_value:
            self.partner_reserved_value = float(self.opponent_ufun(offer)) / 2

        # update rational_outcomes by removing the outcomes that are below the reservation value of the opponent
        # Watch out: if the reserved value decreases, this will not add any outcomes.
        rational_outcomes = self.rational_outcomes = [
            _
            for _ in self.rational_outcomes
            if self.opponent_ufun(_) > self.partner_reserved_value
        ]


# if you want to do a very small test, use the parameter small=True here. Otherwise, you can use the default parameters.
if __name__ == "__main__":
    from helpers.runner import run_a_tournament

    run_a_tournament(AwesomeNegotiator, small=True)
