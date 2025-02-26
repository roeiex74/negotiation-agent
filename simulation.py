from negmas import (
    make_issue,
    SAOMechanism,
    TimeBasedConcedingNegotiator,
    MappingUtilityFunction,
)
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun
from negmas.preferences.value_fun import LinearFun, IdentityFun, AffineFun
import matplotlib.pyplot as plt

import random  # for generating random ufuns
from anl.anl2024.negotiators import Conceder, Boulware, NashSeeker
from helpers import runner

# from hard_chaos import HardChaosNegotiator
random.seed(0)  # for reproducibility

# session = SAOMechanism(outcomes=10, n_steps=100)

# create negotiation agenda (issues)
issues = [
    make_issue(name="price", values=10),
    make_issue(name="quantity", values=(1, 11)),
    make_issue(name="delivery_time", values=10),
]
session = SAOMechanism(issues=issues, n_steps=5000)
# define buyer and seller utilities
seller_utility = LUFun(
    values=[IdentityFun(), LinearFun(0.2), AffineFun(-1, bias=9.0)],
    outcome_space=session.outcome_space,
)

buyer_utility = LUFun(
    values={
        "price": AffineFun(-1, bias=9.0),
        "quantity": LinearFun(0.2),
        "delivery_time": IdentityFun(),
    },
    outcome_space=session.outcome_space,
)
seller_utility = seller_utility.scale_max(1.0)
buyer_utility = buyer_utility.scale_max(1.0)

session.add(
    TimeBasedConcedingNegotiator(name="buyer", offering_curve="boulware"),
    ufun=buyer_utility,
)
session.add(
    TimeBasedConcedingNegotiator(name="seller", offering_curve="boulware"),
    ufun=seller_utility,
)
session.run()
session.plot(ylimits=(0.0, 1.01), show_reserved=False)
plt.show()

# runner.run_a_tournament(HardChaosNegotiator)
