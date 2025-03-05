from negmas.sao import SAOMechanism
from negmas.sao.negotiators import SAONegotiator
from negmas.preferences import LinearAdditiveUtilityFunction
from negmas.common import MechanismRound

# Create a simple negotiation scenario
scenario = make_random_scenario(n_issues=3, n_points=10)

# Create utility functions for two agents
ufun1 = LinearAdditiveUtilityFunction.random(scenario.issue_values)
ufun2 = LinearAdditiveUtilityFunction.random(scenario.issue_values)

# Initialize your agent and a simple opponent (e.g., TimeDependentNegotiator)
agent1 = AwesomeNegotiator(name="MyAgent", ufun=ufun1)
agent2 = SAONegotiator(name="Opponent", ufun=ufun2)

# Run the negotiation mechanism
mechanism = SAOMechanism(negotiators=[agent1, agent2], scenario=scenario)
mechanism.run(verbosity=2)

# Print results
print("Final Agreement:", mechanism.agreement)
print(
    f"{agent1.name} Utility:",
    agent1.ufun(mechanism.agreement) if mechanism.agreement else None,
)
print(
    f"{agent2.name} Utility:",
    agent2.ufun(mechanism.agreement) if mechanism.agreement else None,
)
