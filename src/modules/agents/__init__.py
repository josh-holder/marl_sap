REGISTRY = {}

from .rnn_agent import RNNAgent
from .rnn_ns_agent import RNNNSAgent
from .rnn_feature_agent import RNNFeatureAgent
from .custom_cnn_const_agent import CustomCNNConstellationAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_ns"] = RNNNSAgent
REGISTRY["rnn_feat"] = RNNFeatureAgent
REGISTRY["custom_cnn_const_agent"] = CustomCNNConstellationAgent