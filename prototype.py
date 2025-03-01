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
from negmas import pareto_frontier, nash_points
from negmas.outcomes import Outcome
from negmas.sao import ResponseType, SAONegotiator, SAOResponse, SAOState
from opponent_behavior_estimator import OpponentBehaviorEstimator
from scipy.optimize import curve_fit


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
        self.fe = e
        self.nash_factor = nash_factor
        self.opp_offer_history = []
        self.my_offer_history = []
        self.opponent_utilities = []
        self.my_utilities = []
        self.opponent_times = []
        self.bidding_mode = 0  # Will be used to switch bidding strategy between different scenarion
        self.final_stages = 0.9
        self.phase = "Early"
        self.is_competitive_mode = False

    def _detect_competitive_negotiation(self):
        """Determine if the negotiation is competitive based on utility trade-offs."""
        if not self.my_sorted_outcomes or not self.opp_sorted_outcomes:
            self.is_competitive_mode = False
            return

        # Utilities at extreme outcomes
        my_best_utility = self.my_sorted_outcomes[-1][0]  # Max utility for me
        opp_at_my_best = self.my_sorted_outcomes[-1][
            1
        ]  # Opponent utility when I maximize
        opp_best_utility = self.opp_sorted_outcomes[-1][
            0
        ]  # Max utility for opponent
        my_at_opp_best = self.opp_sorted_outcomes[-1][
            1
        ]  # My utility when opponent maximizes

        # Calculate trade-off differences
        opp_utility_gain = opp_best_utility - opp_at_my_best
        my_utility_loss = my_best_utility - my_at_opp_best

        # Competitive metric: relative trade-off ratio
        # if my_utility_loss != 0:  # Avoid division by zero
        #     tradeoff_ratio = opp_utility_gain / my_utility_loss
        #     is_tradeoff_significant = tradeoff_ratio > 2
        # else:
        #     is_tradeoff_significant = (
        #         opp_utility_gain > 0.2
        #     )  # Fallback to absolute gain
        is_tradeoff_significant = (opp_utility_gain - my_utility_loss) > 0.2
        # Alternative metric: distance from Nash point (if available)
        # is_far_from_nash = False
        # if self.nash_outcome:
        #     nash_my_util = self.my_nash_utility
        #     nash_opp_util = self.opponent_ufun(self.nash_outcome)
        #     dist_my_best = (
        #         (my_best_utility - nash_my_util) ** 2
        #         + (opp_at_my_best - nash_opp_util) ** 2
        #     ) ** 0.5
        #     dist_opp_best = (
        #         (my_at_opp_best - nash_my_util) ** 2
        #         + (opp_best_utility - nash_opp_util) ** 2
        #     ) ** 0.5
        #     is_far_from_nash = max(dist_my_best, dist_opp_best) > 0.3

        # Set competitive mode: significant trade-off or far from Nash
        # self.is_competitive_mode = is_tradeoff_significant or is_far_from_nash
        self.is_competitive_mode = is_tradeoff_significant
        # short negotiations considered competitive
        if self.nmi.n_steps is not None and self.nmi.n_steps <= 50:
            self.is_competitive_mode = True

    def on_preferences_changed(self, changes):
        if self.ufun is None:
            return

        # Utility functions for self and opponent
        ufuns = (self.ufun, self.opponent_ufun)

        # Enumerate or sample all possible outcomes
        all_outcomes = list(self.nmi.outcome_space.enumerate_or_sample())

        # Initialize rational outcomes (above my reserved value)
        self.rational_outcomes = [
            outcome
            for outcome in all_outcomes
            if self.ufun(outcome) > self.ufun.reserved_value
        ]

        self.partner_reserved_value = self.ufun.reserved_value

        # Compute Pareto frontier
        frontier_utils, frontier_indices = pareto_frontier(
            ufuns, self.rational_outcomes
        )
        self.frontier_outcomes = [
            self.rational_outcomes[i] for i in frontier_indices
        ]
        self.my_frontier_utils = [utils[0] for utils in frontier_utils]
        self.opp_frontier_utils = [utils[1] for utils in frontier_utils]

        # Filter frontier outcomes above my reserved value
        self.rational_frontier_outcomes = [
            outcome
            for my_util, outcome in zip(
                self.my_frontier_utils, self.frontier_outcomes
            )
            if my_util >= self.ufun.reserved_value
        ]

        # Compute Nash bargaining solution
        calc_nash_points = nash_points(ufuns, frontier_utils)
        self.my_nash_utility = (
            calc_nash_points[0][0][0]
            if calc_nash_points
            else 0.5 * (float(self.ufun.max()) + self.ufun.reserved_value)
        )
        self.nash_outcome = (
            self.frontier_outcomes[calc_nash_points[0][1]]
            if calc_nash_points
            else None
        )
        if self.nash_outcome:
            self.nash_my_util = self.ufun(self.nash_outcome)
            self.nash_opp_util = self.opponent_ufun(self.nash_outcome)
        else:
            self.nash_my_util = None
            self.nash_opp_util = None
        # Compute outcomes sorted by utilities for competitive mode detection
        self.my_sorted_outcomes = sorted(
            [
                (self.ufun(o), self.opponent_ufun(o), o)
                for o in all_outcomes
                if self.ufun(o) > 0 and self.opponent_ufun(o) > 0
            ],
            key=lambda x: x[0],  # Sort by my utility
        )
        self.best_offer = self.my_sorted_outcomes[-1][-1]
        self.opp_sorted_outcomes = sorted(
            [
                (self.opponent_ufun(o), self.ufun(o), o)
                for o in all_outcomes
                if self.ufun(o) > 0 and self.opponent_ufun(o) > 0
            ],
            key=lambda x: x[0],  # Sort by opponent utility
        )

        # Detect competitive negotiation
        self._detect_competitive_negotiation()

        return super().on_preferences_changed(changes)

    def determine_negotiation_phase(self, state: SAOState) -> str:
        """
        Determines the current phase of negotiation based on relative time
        and adjusts `e` accordingly.

        Returns:
            str: One of ["Early", "Middle", "Final"] representing the negotiation phase.
        """
        # Compute the average step duration (avoid division by zero)
        avg_step_duration = (
            0.0001 if state.step == 0 else state.relative_time / state.step
        )

        # Estimate the final negotiation time based on step size
        self.estimated_final_time = (
            1.0 // avg_step_duration
        ) * avg_step_duration

        # Determine phase
        if state.relative_time < 0.3:
            self.phase = "Early"
        elif state.relative_time < self.final_stages:
            self.phase = "Middle"
        else:
            self.phase = "Final"

        # Dynamic Exponent Adjustment with Bounds
        if self.phase == "Early":
            self.e = self.fe  # Firm stance (e.g., 15)
        elif self.phase == "Middle":
            asp = aspiration_function(
                state.relative_time / self.estimated_final_time,
                1.0,
                self.ufun.reserved_value,
                self.fe,
            )
            adjustment = (1.0 - asp) * 10  # Controlled increase
            self.e = self.fe + adjustment
            if not self.is_opponent_conceding():
                self.e = min(
                    self.e * 1.1, 25.0
                )  # Push higher if opponent is stubborn
            elif self.is_nash_seeking():
                self.e = max(
                    self.e * 0.95, 10.0
                )  # Slightly concede if Nash-seeking
            self.e = min(max(self.e, 5.0), 25.0)  # Enforce bounds
        elif self.phase == "Final":
            if not self.is_opponent_conceding():
                self.e = max(self.fe * 0.85, 5.0)  # Firm but flexible
            else:
                self.e = max(self.fe * 0.9, 7.0)  # Gradual concession

    def update_opponent_offers(self, state: SAOState):
        """Updates the current offer history, opponent utilitie per offer and also
        Finds the current best received offer from the opponent for later use.
        """
        offer = state.current_offer
        if offer is not None:
            self.opp_offer_history.append(state.current_offer)
            self.opponent_utilities.append(self.opponent_ufun(offer))
            self.opponent_times.append(state.relative_time)

    def __call__(self, state: SAOState) -> SAOResponse:
        assert self.ufun and self.opponent_ufun
        self.update_opponent_offers(state)
        self.nmi.log_info(
            self.id,
            dict(
                turn="Opponent OFFER",
                my_util=f"{self.ufun(state.current_offer)}",
                opponent_util=f"{self.opponent_ufun(state.current_offer)}",
            ),
        )
        if self.acceptance_strategy(state):

            return SAOResponse(ResponseType.ACCEPT_OFFER, state.current_offer)

        if not self.rational_outcomes:
            return SAOResponse(ResponseType.REJECT_OFFER, self.best_offer)

        self.determine_negotiation_phase(state)
        one_step = (
            0.0001 if state.step == 0 else state.relative_time / state.step
        )

        # Adjust strategy based on opponent behavior
        opponent_is_conceding = self.is_opponent_conceding()
        opponent_is_nash_seeking = self.is_nash_seeking()

        if self.phase == "Final" or state.relative_time + 3 * one_step > 1.0:
            # Final stages: Optimize offer like Shochan
            if self.opponent_utilities:
                opp_min = min(self.opponent_utilities)
                opp_max = max(self.opponent_utilities)
                opp_target = opp_max - (opp_max - opp_min) * (
                    state.relative_time / max(self.opponent_times)
                )
            else:
                opp_target = 0.0  # Default if no offers yet

            # Initialize with a safe fallback ensuring reservation value
            selected_outcome = (
                self.best_offer
                if self.ufun(self.best_offer) > self.ufun.reserved_value
                else None
            )
            my_best_util = self.ufun.reserved_value + 0.05  # Minimum threshold

            # Iterate over all outcomes to maximize utility like Shochan
            for opp_util, my_util, outcome in reversed(
                self.opp_sorted_outcomes
            ):
                if (
                    opp_util >= opp_target
                    and my_util >= self.ufun.reserved_value
                ):
                    if my_util > my_best_util:
                        my_best_util = my_util
                        selected_outcome = outcome
                elif opp_util < opp_target:
                    break  # Stop when below target

            # Refine based on opponent behavior
            if opponent_is_conceding and self.opp_offer_history:
                best_received = max(self.opp_offer_history, key=self.ufun)
                if self.ufun(best_received) > my_best_util:
                    selected_outcome = best_received
                    my_best_util = self.ufun(best_received)
            elif (
                opponent_is_nash_seeking
                and self.nash_outcome
                and self.ufun(self.nash_outcome) > self.ufun.reserved_value
            ):
                if self.ufun(self.nash_outcome) > my_best_util:
                    selected_outcome = self.nash_outcome

            # Fallback if no optimized outcome found
            if not selected_outcome:
                selected_outcome = (
                    self.best_offer
                    if self.ufun(self.best_offer) > self.ufun.reserved_value
                    else self.nash_outcome
                )
        else:
            my_at_opp_best = self.opp_sorted_outcomes[-1][1]
            aspiration_minimum = (
                max(self.ufun.reserved_value, self.my_nash_utility)
                if self.is_competitive_mode and self.nash_outcome
                else max(self.ufun.reserved_value, my_at_opp_best)
            )
            if opponent_is_nash_seeking and self.nash_outcome:
                # Bias toward Nash in early/middle if opponent seeks it
                aspiration_minimum = max(
                    aspiration_minimum, self.my_nash_utility * 0.95
                )  # Slightly below Nash
            elif not opponent_is_conceding:
                # Raise aspiration if opponent isn’t conceding
                aspiration_minimum = max(
                    aspiration_minimum, self.my_nash_utility or my_at_opp_best
                )

            aspiration_target = aspiration_function(
                state.relative_time, 1.0, aspiration_minimum, self.e
            )
            max_index = len(self.my_sorted_outcomes) - 1
            current_index = max_index
            while current_index > 0:
                next_index = current_index - 1
                if self.my_sorted_outcomes[next_index][0] >= aspiration_target:
                    current_index = next_index
                else:
                    break
            selected_outcome = self.my_sorted_outcomes[current_index][2]
        self.my_offer_history.append(selected_outcome)
        self.my_utilities.append(float(self.ufun(selected_outcome)))
        self.nmi.log_info(
            self.id,
            dict(
                turn="Suggesting Offer",
                # my_offer=f"{selected_outcome}",
                my_util=f"{self.ufun(selected_outcome)}",
                opponent_util=f"{self.opponent_ufun(selected_outcome)}",
                # is_conceding=(
                #     "Concider" if opponent_is_conceding else "not conceding"
                # ),
                # is_nash_seeker=(
                #     "Nash seeker"
                #     if opponent_is_nash_seeking
                #     else "not nash seeker"
                # ),
                # competitive=(
                #     "Competitive"
                #     if self.is_competitive_mode
                #     else "Not competitive"
                # ),
                current_phase=self.phase,
            ),
        )
        return SAOResponse(ResponseType.REJECT_OFFER, selected_outcome)

    def is_opponent_conceding(self, last_n=3, min_threshold=0.02):
        """
        Detects whether the opponent is conceding in the last 'N' rounds.
        - last_n: Number of recent rounds to analyze.
        - min_threshold: Minimum utility drop to count as a real concession.
        """

        if len(self.opponent_utilities) < last_n + 1:
            return False  # Not enough data to decide

        # Compute utility differences
        recent_drops = [
            self.opponent_utilities[i] - self.opponent_utilities[i + 1]
            for i in range(-last_n - 1, -1)
        ]

        # Count how many times the opponent actually conceded
        num_concessions = sum(
            1 for drop in recent_drops if drop > min_threshold
        )

        # Compute average concession amount
        avg_concession = sum(drop for drop in recent_drops if drop > 0) / max(
            num_concessions, 1
        )

        # If the opponent is conceding frequently and significantly, return True
        return (
            num_concessions >= last_n * 0.4 and avg_concession > min_threshold
        )

    def is_nash_seeking(self, proximity_threshold=0.05, n_recent=3):
        """
        Determines if the opponent is a Nash seeker by checking if their recent offers
        are close to the Nash bargaining solution for both parties.
        """
        # Check if we have enough data and a Nash outcome
        if not self.nash_outcome or len(self.opp_offer_history) < n_recent:
            return False

        last_n_offers = self.opp_offer_history[-n_recent:]

        for offer in last_n_offers:
            my_util = self.ufun(offer)
            opp_util = self.opponent_ufun(offer)

            # Check if both utilities are within threshold of Nash utilities
            if (
                abs(my_util - self.nash_my_util) > proximity_threshold
                or abs(opp_util - self.nash_opp_util) > proximity_threshold
            ):
                return False  # Offer is too far from Nash point

        return True  # All recent offers are near the Nash point

    def acceptance_strategy(self, state: SAOState) -> bool:
        """
        Determines whether to accept the opponent's offer based on:
        - Negotiation phase (self.phase)
        - Opponent behavior (concessions)

        """
        offer = state.current_offer
        best_received_offer = self.best_opponent_offer()
        if offer is None:
            return False  # No offer, cannot accept.

        # Compute time left in the negotiation
        avg_step_duration = (
            0.0001 if state.step == 0 else state.relative_time / state.step
        )
        time_left = 1.0 - state.relative_time

        # Determine if opponent is conceding
        is_opponent_conceding = self.is_opponent_conceding()

        # **Set Aspiration Level Based on Phase**
        if self.phase == "Early":
            # Hold firm in early rounds
            border = self.ufun.reserved_value
        elif self.phase == "Middle":
            # Gradual concession
            border = aspiration_function(
                state.relative_time, 1.0, self.ufun.reserved_value, self.e
            )
        else:  # Final Phase
            # Define when the final phase starts based on the estimated final negotiation time
            final_phase_start_time = (
                self.final_stages * self.estimated_final_time
            )

            if state.relative_time >= final_phase_start_time:
                # Compute ShoChan-style aspiration decay
                initial_asp = aspiration_function(
                    final_phase_start_time / self.estimated_final_time,
                    1.0,
                    self.ufun.reserved_value,
                    self.e,
                )

                # Quadratic aspiration decay function
                decay_factor = (
                    self.estimated_final_time - final_phase_start_time
                ) ** 2
                concession_speed = (
                    initial_asp - self.ufun.reserved_value
                ) / decay_factor
                xd = (
                    state.relative_time / self.estimated_final_time
                ) - final_phase_start_time
                adjusted_asp = initial_asp - (concession_speed * xd * xd)
                border = max(
                    self.ufun.reserved_value, adjusted_asp
                )  # Ensure no over-concession
            else:
                border = self.ufun.reserved_value

        # Compute final aspiration threshold
        myasp = aspiration_function(state.relative_time, 1.0, border, self.e)

        # **Final Adjustments Based on Time Left**
        if time_left < avg_step_duration * 3:
            if not is_opponent_conceding:
                myasp = (
                    max(
                        self.ufun.reserved_value,
                        self.ufun(best_received_offer),
                    )
                    + 0.05
                )
            else:
                myasp = max(
                    self.ufun.reserved_value,
                    self.ufun(best_received_offer),
                )

        # **Last Step Handling**
        if state.relative_time + avg_step_duration >= 1.0:
            myasp = max(
                self.ufun.reserved_value,
                self.ufun(best_received_offer) - 0.05,
            )

        # Log for debug
        self.nmi.log_debug(
            self.id,
            dict(
                turn="Checking Acceptance",
                aspiration=f"{myasp}",
                reservation_value=f"{self.ufun.reserved_value}",
            ),
        )
        return float(self.ufun(offer)) >= myasp

    # def predict_concession_price(self, state: SAOState) -> Outcome:
    #     """Predicts the opponent's next concession offer using linear regression."""

    #     if state.step < 3:
    #         return self.best_offer  # Not enough data for regression

    #     # Extract time and opponent utilities
    #     times = np.array(
    #         [
    #             i / len(self.opp_offer_history)
    #             for i in range(len(self.opp_offer_history))
    #         ]
    #     )
    #     opponent_utils = np.array(
    #         [self.opponent_ufun(offer) for offer in self.opp_offer_history]
    #     )

    #     # Ensure we have enough variation to fit a model
    #     if len(times) < 3 or np.all(opponent_utils == opponent_utils[0]):
    #         return self.best_offer  # Not enough change in offers

    #     # Linear regression model: Opponent utility as a function of time
    #     def linear_model(t, a, b):
    #         return a * t + b

    #     # Fit the model
    #     try:
    #         (a, b), _ = curve_fit(linear_model, times, opponent_utils)
    #     except Exception as e:
    #         print(f"Curve fitting failed: {e}")
    #         return self.best_offer  # Fallback to best known offer

    #     # Predict the opponent's utility at the next time step
    #     predicted_utility = linear_model(
    #         state.relative_time + (1 / (state.step + 1)), a, b
    #     )

    #     # Select the closest rational outcome based on the predicted opponent utility
    #     best_offer = min(
    #         self.rational_outcomes,
    #         key=lambda o: abs(self.opponent_ufun(o) - predicted_utility),
    #     )

    #     return best_offer

    def best_opponent_offer(self) -> Outcome:
        if not self.opp_offer_history:
            return self.ufun.best()  # Fallback to best known offer
        return max(self.opp_offer_history, key=lambda h: self.ufun(h))


# if you want to do a very small test, use the parameter small=True here. Otherwise, you can use the default parameters.
if __name__ == "__main__":
    from helpers.runner import run_a_tournament

    run_a_tournament(AwesomeNegotiator, small=True)
