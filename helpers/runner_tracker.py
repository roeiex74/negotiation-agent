import pandas as pd
import matplotlib.pyplot as plt
from anl.anl2024 import anl2024_tournament
from anl.anl2024.negotiators import Conceder, Boulware, NashSeeker
from hard_chaos import HardChaosNegotiator
from negmas.helpers import humanize_time, unique_name
import time


def analyze_adaptation(
    TestedNegotiator, n_repetitions=5, n_scenarios=5, nologs=False
):
    """
    Runs a small tournament with a specific negotiator and tracks how it performs against different strategies.
    """
    start = time.perf_counter()
    name = (
        unique_name(
            f"test{TestedNegotiator().type_name.split('.')[-1]}", sep=""
        )
        if not nologs
        else None
    )
    results = []

    # Define competitors to test adaptation
    competitors = [Conceder, Boulware, NashSeeker]

    for Opponent in competitors:
        print(f"Testing {TestedNegotiator.__name__} vs {Opponent.__name__}...")

        # Run a small tournament between the agent and the opponent
        scores = anl2024_tournament(
            competitors=(TestedNegotiator, Opponent),
            n_scenarios=n_scenarios,
            n_outcomes=1000,
            n_repetitions=n_repetitions,
            njobs=0,
            verbosity=1,
            plot_fraction=0.5,
            nologs=nologs,
        ).final_scores

        # Store the results
        for row in scores.itertuples():
            results.append(
                {
                    "agent": row.strategy,
                    "opponent": Opponent.__name__,
                    "score": row.score,
                }
            )
    print(f"Finished in {humanize_time(time.perf_counter() - start)}")
    # Convert results to DataFrame
    df_results = pd.DataFrame(results)

    # Display results
    print("\nPerformance Summary:")
    print(df_results)

    # Plot performance per opponent
    plt.figure(figsize=(10, 6))
    for opponent in df_results["opponent"].unique():
        subset = df_results[df_results["opponent"] == opponent]
        plt.plot(
            subset["agent"],
            subset["score"],
            marker="o",
            label=f"vs {opponent}",
        )

    plt.xlabel("Agent")
    plt.ylabel("Average Utility Score")
    plt.title("Agent Performance Against Different Opponents")
    plt.legend()
    plt.grid(True)
    plt.show()
