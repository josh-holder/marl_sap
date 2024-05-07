REGISTRY = {}

from .classic_selectors import MultinomialActionSelector, EpsilonGreedyActionSelector, SoftPoliciesSelector
from .bet_selectors import ContinuousActionSelector
from .sap_selectors import SequentialAssignmentProblemSelector, EpsilonGreedySAPTestActionSelector
from filtered_sap_selectors import FilteredSAPActionSelector, FilteredEpsGrSAPTestActionSelector

REGISTRY["continuous"] = ContinuousActionSelector
REGISTRY["multinomial"] = MultinomialActionSelector
REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector
REGISTRY["soft_policies"] = SoftPoliciesSelector
REGISTRY["sap"] = SequentialAssignmentProblemSelector
REGISTRY["epsilon_greedy_sap_test"] = EpsilonGreedySAPTestActionSelector
REGISTRY["real_const_sap"] = FilteredSAPActionSelector
REGISTRY["real_const_epsgr_sap_test"] = FilteredEpsGrSAPTestActionSelector