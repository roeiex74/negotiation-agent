"""
**Submitted to ANAC 2024 Automated Negotiation League**
*Team* Group 10
*Authors* Roei Rahamim - roei.rahamim@runi.ac.il, Hadar Manasherov - hadar.manasherov@runi.ac.il

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 ANL competition.
"""

import random
import numpy as np
from negmas.outcomes import Outcome
from negmas.sao import ResponseType, SAONegotiator, SAOResponse, SAOState


def aspiration_function(t, mx, rv, e):
    """
    rv - reservation value
    mx - maximum utility
    e - exponent for defining the rate of concessions
    """
    return (mx - rv) * (1.0 - np.power(t, e)) + rv


class Group10(SAONegotiator):
    """
    Your agent code. This is the ONLY class you need to implement
    """

    def __init__(
        self,
        preferences=None,
        ufun=None,
        name=None,
        parent=None,
        owner=None,
        id=None,
        type_name=None,
        can_propose=True,
        **kwargs,
    ):
        super().__init__(
            preferences,
            ufun,
            name,
            parent,
            owner,
            id,
            type_name,
            can_propose,
            **kwargs,
        )
        self.opponent_offers = []
        self.fake_reservation_value = None
        self.opponent_reserved_value = 0.1
        self.best_offer = None

    rational_outcomes = tuple()

    partner_reserved_value = 0

    def on_preferences_changed(self, changes):
        """
        Called when preferences change. In ANL 2024, this is equivalent with initializing the agent.

        Remarks:
            - Can optionally be used for initializing your agent.
            - We use it to save a list of all rational outcomes.

        """
        # If there a no outcomes (should in theory never happen)
        assert self.ufun is not None
        assert self.opponent_ufun is not None

        self.rational_outcomes = [
            _
            for _ in self.nmi.outcome_space.enumerate_or_sample()  # enumerates outcome space when finite, samples when infinite
            if self.ufun(_) > self.ufun.reserved_value
        ]
        self.best_offer = self.ufun.best()

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

        # if self.ufun(offer) > (2 * self.ufun.reserved_value):
        if self.ufun(offer) > (2 * self.fake_reservation_value):
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

    def on_partner_proposal(self, state, partner_id, offer):
        return super().on_partner_proposal(state, partner_id, offer)

    def on_negotiation_end(self, state: SAOState):
        """Save opponent reservation price for next negotiations."""
        if self.opponent_offers:
            avg_reservation = np.mean(self.opponent_offers)
            print(f"Final estimated reservation price: {avg_reservation}")


# if you want to do a very small test, use the parameter small=True here. Otherwise, you can use the default parameters.
if __name__ == "__main__":
    from helpers.runner import run_a_tournament

    run_a_tournament(Group10, small=True)
