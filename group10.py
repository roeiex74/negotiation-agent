"""
**Submitted to ANAC 2024 Automated Negotiation League**
*Team* type your team name here: **Group 10**
*Authors* type your team member names with their emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 ANL competition.
"""

import random
import numpy as np
from negmas import pareto_frontier, nash_points
from negmas.outcomes import Outcome
from negmas.sao import ResponseType, SAONegotiator, SAOResponse, SAOState


def aspiration_function(t, mx, rv, e):
    """
    Time-dependent aspiration function..
    Determines the minimum acceptable utility at a given time step based on the agent's:
    reserved value, maximum utility, and negotiation time elapsed.
    """
    return (mx - rv) * (1.0 - np.power(t, e)) + rv


def average_step_time(state: SAOState) -> float:
    """
    Calculates the average step time to determine the last two steps
    (to estimate the negotiation progress- help determine when the final stages of the negotiation begins).
    """
    return state.relative_time / max(1, state.step)


class Group10(SAONegotiator):
    rational_outcomes = tuple()
    partner_reserved_value = 0
    time_dependent_threshold = 0.95

    def __init__(
        self,
        *args,
        e: float = 18.5,
        debug=True,
        **kwargs,
    ):
        """
        Initialize the negotiation agent with adjustable parameters:
        -> e:
            Determines initial firmness in offering strategy.
        -> min_unique_utilities:
            Ensures diverse bid selection.
        """
        super().__init__(*args, **kwargs)
        self.e = e
        self.fe = e
        self.opp_offer_history = []
        self.my_offer_history = []
        self.opponent_utilities = []
        self.my_utilities = []
        self.opponent_times = []
        self.final_stages = 0.95
        self.phase = "Early"
        self.is_competitive_mode = False
        self.epsilon = 1e-8  # A small buffer to avoid division by zero in ratio computations
        self.debug = debug

    def _detect_competitive_negotiation(self):
        """
        Determine if the negotiation is competitive based on utility trade-offs.
        (If the opponent benefits more from their best deals, suggest a competitive scenario requiring strategic adjustments)
        """
        if not self.my_sorted_outcomes or not self.opp_sorted_outcomes:
            self.is_competitive_mode = False
            return

        # Utilities at extreme outcomes
        my_best_utility = self.my_sorted_outcomes[-1][0]  # Max utility
        opp_at_my_best = self.my_sorted_outcomes[-1][
            1
        ]  # Opponent utility from the max utility for us out of his outcomes
        opp_best_utility = self.opp_sorted_outcomes[-1][
            0
        ]  # Max utility for opponent
        my_at_opp_best = self.opp_sorted_outcomes[-1][
            1
        ]  # Utility when opponent maximizes

        # Calculate trade-off differences
        opp_utility_gain = opp_best_utility - opp_at_my_best
        my_utility_loss = my_best_utility - my_at_opp_best

        is_tradeoff_significant = (opp_utility_gain - my_utility_loss) > 0.2

        self.is_competitive_mode = is_tradeoff_significant

        #  Short negotiations considered competitive
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
            adjustment = (1.0 - asp) * 25  # Controlled increase
            self.e = self.fe + adjustment
            if not self.is_opponent_conceding():
                self.e = min(
                    self.e * 1.05, 25.0
                )  # Push higher if opponent is stubborn
            elif self.is_nash_seeking():
                self.e = max(
                    self.e * 0.9, 10.0
                )  # Slightly concede if Nash-seeking
            self.e = min(max(self.e, 5.0), 25.0)  # Enforce bounds
        elif self.phase == "Final":
            if not self.is_opponent_conceding():
                self.e = max(self.fe * 0.95, 5.0)  # Firm but flexible
            else:
                self.e = max(self.fe * 0.9, 7.0)  # Gradual concession

    def update_partner_reserved_value(self, state: SAOState) -> None:
        """Updates the current offer history, opponent utilitie per offer and also
        Finds the current best received offer from the opponent for later use.
        """
        offer = state.current_offer
        if offer is not None:
            self.opp_offer_history.append(state.current_offer)
            self.opponent_utilities.append(self.opponent_ufun(offer))
            self.opponent_times.append(state.relative_time)

    def bidding_strategy(self, state: SAOState) -> Outcome | None:
        """
        Returns: The counter offer as Outcome.
        """
        selected_outcome = None
        # The opponent's ufun can be accessed using self.opponent_ufun, which is not used yet.
        one_step = (
            0.0001 if state.step == 0 else state.relative_time / state.step
        )

        # Adjust strategy based on opponent behavior
        opponent_is_conceding = self.is_opponent_conceding()
        opponent_is_nash_seeking = self.is_nash_seeking()

        if self.phase == "Final":
            opponent_is_stubborn = self.is_opponent_stubborn()
            if opponent_is_stubborn and self.debug:
                self.nmi.log_info(
                    self.id, dict(type="Stubborn Opponent found")
                )
            if state.relative_time + one_step >= 1.0:
                # Make a last attempt for a deal
                selected_outcome = self.final_step_offer()
                if self.debug:
                    self.nmi.log_info(
                        "Final_Offer_strategies",
                        dict(
                            phase=self.phase,
                            opponent_type="RegularLastStep",
                            current_outcome=f"{selected_outcome}",
                            my_util=f"{self.ufun(selected_outcome)}",
                            opp_util=f"{self.opponent_ufun(selected_outcome)}_FinalOffer########",
                        ),
                    )
            elif (
                state.relative_time + 3 * one_step > 1.0
                or not opponent_is_stubborn
            ):
                # Optimize offer based on opponent offers diversity and estimated aspirations
                if self.opponent_utilities:
                    opp_min = min(self.opponent_utilities)
                    opp_max = max(self.opponent_utilities)
                    if opp_max - opp_min < 0.2:  # Lack of diversity
                        opp_target = np.mean(self.opponent_utilities) / 5
                    else:
                        opp_target = opp_max - (opp_max - opp_min) * (
                            state.relative_time / max(self.opponent_times)
                        )
                else:
                    opp_target = 0.0

                selected_outcome = (
                    self.best_offer
                    if self.ufun(self.best_offer) > self.ufun.reserved_value
                    else None
                )
                my_best_util = (
                    self.ufun.reserved_value + 0.1
                )  # Minimum gain threshold

                for opp_util, my_util, outcome in reversed(
                    self.opp_sorted_outcomes
                ):
                    if (
                        opp_util >= opp_target
                        and my_util >= my_best_util
                        and opp_util / (my_util + self.epsilon)
                        <= 1.3  # Utility ratio check
                    ):
                        if my_util > my_best_util:
                            my_best_util = my_util
                            selected_outcome = outcome
                    elif opp_util < opp_target:
                        break
                if self.debug:
                    self.nmi.log_info(
                        "Final_Offer_strategies",
                        dict(
                            phase=self.phase,
                            opponent_type="Regular",
                            current_outcome=f"{selected_outcome}",
                            my_util=f"{self.ufun(selected_outcome)}",
                            opp_util=f"{self.opponent_ufun(selected_outcome)}########",
                        ),
                    )
                if opponent_is_conceding and self.opp_offer_history:
                    best_received = max(self.opp_offer_history, key=self.ufun)

                    # Ensure we still get a profitable deal
                    if self.ufun(best_received) > my_best_util:
                        selected_outcome = best_received
                        my_best_util = self.ufun(best_received)

                    # Check ratio between utilities
                    offer_ratio = self.opponent_ufun(
                        selected_outcome
                    ) / self.ufun(selected_outcome)

                    if offer_ratio > 1:  # Opponent is still gaining too much
                        # Search for an offer that is slightly more balanced
                        for opp_util, my_util, outcome in reversed(
                            self.opp_sorted_outcomes
                        ):
                            if (
                                my_util > self.ufun.reserved_value + 0.1
                                and my_util / opp_util >= 1.2
                            ):  # Ensure getting a better deal
                                selected_outcome = outcome
                                break
                    if self.debug:
                        self.nmi.log_info(
                            "Final_Offer_strategies",
                            dict(
                                phase=self.phase,
                                opponent_type="Conceding",
                                current_outcome=f"{selected_outcome}",
                                my_util=f"{self.ufun(selected_outcome)}",
                                opp_util=f"{self.opponent_ufun(selected_outcome)}########",
                            ),
                        )

                elif (
                    opponent_is_nash_seeking
                    and self.nash_outcome
                    and self.ufun(self.nash_outcome) > self.ufun.reserved_value
                ):
                    if self.ufun(self.nash_outcome) > my_best_util:
                        selected_outcome = self.nash_outcome
                    if self.debug:
                        self.nmi.log_info(
                            "Final_Offer_strategies",
                            dict(
                                phase=self.phase,
                                opponent_type="Nash Seeking with nash outcome",
                                current_outcome=f"{selected_outcome}",
                                my_util=f"{self.ufun(selected_outcome)}",
                                opp_util=f"{self.opponent_ufun(selected_outcome)}########",
                            ),
                        )
                if not selected_outcome:
                    selected_outcome = (
                        self.best_offer
                        if self.ufun(self.best_offer)
                        > self.ufun.reserved_value
                        else self.nash_outcome
                    )
            else:
                # Stubborn opponent, not in last 3 steps
                progress = (state.relative_time - self.final_stages) / (
                    1.0 - self.final_stages
                )
                progress = max(0, min(progress, 1))

                my_max_util = self.ufun(self.best_offer)
                my_min_util = self.ufun.reserved_value + 0.1
                my_target = max(
                    my_max_util - (my_max_util - my_min_util) * progress,
                    my_min_util,
                )

                beneficial_offers = [
                    (opp_u, my_u, outcome)
                    for opp_u, my_u, outcome in self.opp_sorted_outcomes
                    if my_u >= my_target
                ]
                if beneficial_offers:
                    beneficial_offers.sort(
                        key=lambda x: x[1], reverse=True
                    )  # Bias toward higher my_util
                    top_offers = beneficial_offers[:5]
                    selected_outcome = random.choice(top_offers)[2]
                else:
                    selected_outcome = self.best_offer

        else:
            my_at_opp_best = self.opp_sorted_outcomes[-1][1]
            aspiration_minimum = (
                max(self.ufun.reserved_value, self.my_nash_utility)
                if self.is_competitive_mode and self.nash_outcome
                else max(self.ufun.reserved_value, my_at_opp_best)
            )
            if opponent_is_nash_seeking and self.nash_outcome:
                # Bias toward Nash if opponent seeks it
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

        return selected_outcome

    def __call__(self, state: SAOState) -> SAOResponse:
        assert self.ufun and self.opponent_ufun
        self.update_partner_reserved_value(state)
        if self.debug:
            self.nmi.log_info(
                self.id,
                dict(
                    turn="Opponent OFFER",
                    my_util=f"{self.ufun(state.current_offer)}",
                    opponent_util=f"{self.opponent_ufun(state.current_offer)}",
                ),
            )

        util_ratio = self.opponent_ufun(state.current_offer) // (
            self.ufun(state.current_offer) + self.epsilon
        )
        if (
            self.acceptance_strategy(state)
            and self.ufun.reserved_value + 0.1
            <= self.ufun(state.current_offer)
            and util_ratio <= 1.4
        ):
            return SAOResponse(ResponseType.ACCEPT_OFFER, state.current_offer)

        if not self.rational_outcomes:
            return SAOResponse(ResponseType.REJECT_OFFER, self.best_offer)

        self.determine_negotiation_phase(state)

        selected_outcome = self.bidding_strategy(state)
        if selected_outcome is None:
            if self.debug:
                self.nmi.log_error(
                    "Error_bidding",
                    dict(
                        msg="Could not match any offer from bidding strategy."
                    ),
                )
            selected_outcome = self.best_offer

        self.my_offer_history.append(selected_outcome)
        self.my_utilities.append(float(self.ufun(selected_outcome)))
        if self.debug:
            self.nmi.log_info(
                self.id,
                dict(
                    turn="Suggesting Offer",
                    my_util=f"{self.ufun(selected_outcome)}",
                    opponent_util=f"{self.opponent_ufun(selected_outcome)}",
                    current_phase=self.phase,
                ),
            )

        return SAOResponse(ResponseType.REJECT_OFFER, selected_outcome)

    def estimate_opponent_reservation_value(self):
        """
        Estimates the opponent's reservation value based on actual offers, avoiding reliance on behavior patterns alone.
        Ensures fairness in final-stage deals and adjusts for risk when opponent's utility is extremely low.
        """

        if not self.opponent_utilities:
            return random.uniform(
                0.2, 0.4
            )  # Stochastic fallback if no data available

        # Step 1: Identify true minimum opponent offer
        opp_min = min(
            self.opponent_utilities
        )  # The lowest opponent offer observed

        opp_recent = (
            self.opponent_utilities[-5:]
            if len(self.opponent_utilities) >= 5
            else self.opponent_utilities
        )

        # Opponent's latest offers trend
        opp_recent_min = min(opp_recent)

        # Step 3: Calculate estimated RV based on "true" concessions
        true_concession_threshold = 0.05  # Minimum utility drop required to consider as a real concession
        true_concession = (
            opp_min < 0.6
            and (opp_recent_min - opp_min) < true_concession_threshold
        )

        # Step 4: Adjust the estimation for fairness and risk mitigation
        if true_concession:
            estimated_rv = (
                opp_min * 0.6
            )  # Slight buffer to avoid exact matching
        else:
            estimated_rv = min(
                opp_min - 0.4, 0.25
            )  # More cautious lowering to prevent bad deals

        # Step 5: Risk-aware final adjustments
        if opp_min < 0.2:
            # Opponent's offers are too low—possibly bluffing
            estimated_rv = opp_min * 0.8
        if self.debug:
            self.nmi.log_info(
                "Estimated_opp_rv", dict(res_value=f"{estimated_rv}#######")
            )
        return estimated_rv

    def is_opponent_conceding(self, window=10, min_trend=0.005, min_drop=0.02):
        """
        Determine if the opponent is conceding based on recent utility values.

        This method analyzes the opponent's recent utility values to detect any concession trends.
        It checks if the average drop between consecutive utility values (trend) over a specified
        window exceeds a minimum threshold, or if the most recent drop is greater than a specified threshold.

        Parameters:
            window (int): The number of recent utility values to consider for analysis (default: 10).
            min_trend (float): The minimum average decrease between consecutive utility values required
                            to infer a concession trend (default: 0.005).
            min_drop (float): The minimum drop between the last two utility values required to infer an
                            immediate concession (default: 0.02).

        Returns:
            bool: True if either the average concession trend or the latest concession drop exceeds
                the specified thresholds, indicating that the opponent is conceding; otherwise, False.
        """
        # Check if there are enough data points to evaluate the concession trend.
        if len(self.opponent_utilities) < window:
            return False

        # Extract the most recent 'window' number of utility values.
        recent_utils = self.opponent_utilities[-window:]
        # Calculate the differences between consecutive utility values.
        # Each difference represents the change from one negotiation step to the next.
        # A positive value indicates a drop in utility

        diffs = [
            recent_utils[i] - recent_utils[i + 1]
            for i in range(len(recent_utils) - 1)
        ]

        # Compute the average concession trend over the recent moves.
        # A higher positive average suggests a consistent pattern of conceding.
        trend = sum(diffs) / len(diffs)  # Positive = concession

        latest_drop = (
            recent_utils[-2] - recent_utils[-1]
            if len(recent_utils) >= 2
            else 0
        )
        # Return True if either:
        # - The average trend exceeds the minimum concession threshold (min_trend), or
        # - The latest concession drop exceeds the minimum drop threshold (min_drop).
        return trend > min_trend or latest_drop > min_drop

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

    def is_opponent_stubborn(
        self,
        phase_threshold=0.7,
        stubbornness_threshold=0.15,
        min_concession_rate=0.05,
    ):
        """
        Determines if the opponent has been very stubborn during the negotiation, especially
        after a specified phase (e.g., 60-70% of negotiation time), based on their offers.

        Args:
            phase_threshold (float): Fraction of negotiation time after which to check .
            stubbornness_threshold (float): Minimum utility improvement to not be stubborn .
            min_concession_rate (float): Minimum rate of concession to not be stubborn .

        Returns:
            bool: True if the opponent is stubborn, False otherwise.
        """
        # Check if there’s enough data
        if not self.opp_offer_history or len(self.opp_offer_history) < 5:
            return False  # Not enough offers to judge stubbornness

        # Calculate utilities of opponent’s offers from your perspective
        opp_offers_utils = [
            self.ufun(offer) for offer in self.opp_offer_history
        ]
        times = np.array(self.opponent_times)

        # Focus on offers after the phase threshold (e.g., 60% or 70%)
        mask = times >= phase_threshold
        if not np.any(mask):
            return False  # Haven’t reached the phase of interest yet

        relevant_utils = np.array(opp_offers_utils)[mask]
        relevant_times = times[mask]

        if len(relevant_utils) < 3:
            return False  # Not enough data in this phase to analyze

        # Check improvement: Compare max utility to initial utility in this phase
        initial_util = relevant_utils[0]
        max_util = np.max(relevant_utils)
        improvement = max_util - initial_util

        if improvement > stubbornness_threshold:
            return False  # Opponent has made a meaningful concession

        # Check concession trend: Use slope of utilities over time
        slope, _ = np.polyfit(relevant_times, relevant_utils, 1)
        if slope > min_concession_rate:
            return False  # Opponent is conceding at a reasonable rate

        # If no significant improvement or concession rate, opponent is stubborn
        return True

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

        # Set Aspiration Level Based on Phase
        if self.phase == "Early":
            # Hold firm in early rounds
            # Calculate my utility and maximum possible utility
            my_util = float(
                self.ufun(offer)
            )  # Utility function evaluates the offer
            max_util = float(self.ufun.max())  # Max utility achievable

            # Early acceptance: high utility in first 30% of negotiation
            if state.relative_time < 0.3 and my_util >= 0.9 * max_util:
                return True
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

            # Compute aspiration decay
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

        # Compute final aspiration threshold
        myasp = aspiration_function(state.relative_time, 1.0, border, self.e)

        # Final Adjustments Based on Time Left
        if time_left < avg_step_duration * 3:
            if not is_opponent_conceding:
                myasp = (
                    max(
                        self.ufun.reserved_value,
                        self.ufun(best_received_offer),
                    )
                    + 0.2
                )
            else:
                myasp = max(
                    self.ufun.reserved_value,
                    self.ufun(best_received_offer),
                )

        # Log for debug
        if self.debug:
            self.nmi.log_debug(
                self.id,
                dict(
                    turn="Checking Acceptance",
                    aspiration=f"{myasp}",
                    reservation_value=f"{self.ufun.reserved_value}",
                ),
            )

        return float(self.ufun(offer)) >= myasp

    def best_opponent_offer(self) -> Outcome:
        """Returns maximal Outcome object from opponent offers during negotiation in terms of our utility."""

        # No offers yet - allow a fallbck
        if not self.opp_offer_history:
            return self.ufun.best()  # Fallback to best known offer
        # Select the max outcome by our utility
        return max(self.opp_offer_history, key=lambda h: self.ufun(h))

    def final_step_offer(self):
        """
        Generates the best possible final step offer by leveraging opponent reservation estimation.
        Ensures fairness while avoiding extreme concessions.
        """

        estimated_rv = self.estimate_opponent_reservation_value()

        # Default to the best known offer in case of uncertainty
        selected_outcome = (
            self.best_offer
            if self.ufun(self.best_offer) > self.ufun.reserved_value
            else None
        )

        my_best_util = (
            self.ufun.reserved_value + 0.1
        )  # Minimum threshold for a fair deal

        for opp_util, my_util, outcome in reversed(self.opp_sorted_outcomes):
            if (
                opp_util >= estimated_rv
                and my_util >= my_best_util
                and opp_util / (my_util + self.epsilon)
                <= 1.5  # Avoid extreme imbalance
            ):
                if my_util > my_best_util and my_util / opp_util <= 1.5:
                    my_best_util = my_util
                    selected_outcome = outcome
            elif opp_util < estimated_rv:
                break  # Stop searching when opponent utilities go too low

        # **Final risk-aware strategy**
        if selected_outcome is None:
            # No good match found, take a calculated risk
            if estimated_rv > 0.4:
                selected_outcome = self.best_offer  # Stick to best outcome
            else:
                selected_outcome = random.choice(
                    self.rational_outcomes
                )  # Offer a reasonable alternative

        return selected_outcome


# if you want to do a very small test, use the parameter small=True here. Otherwise, you can use the default parameters.
if __name__ == "__main__":
    from helpers.runner import run_a_tournament

    run_a_tournament(Group10, small=True)
