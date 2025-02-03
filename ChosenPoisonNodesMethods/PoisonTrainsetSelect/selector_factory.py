from enum import Enum
from .random_selector import RandomSelector
from .degree_selector import DegreeBasedSelector
from .pagerank_selector import PageRankBasedSelector
from .llm_selector import LLMBasedSelector
from .cluster_selector import ClusterBasedSelector
from .selector_config import *


class SelectorType(Enum):
    RANDOM = "random"
    DEGREE = "degree"
    PAGERANK = "pagerank"
    LLM = "llm"
    CLUSTER = "cluster"

    @classmethod
    def from_string(cls, value: str) -> 'SelectorType':
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(f"Unsupported selector type: {value}")


class PoisonNodeSelectorFactory:
    @staticmethod
    def create_selector(selector_type: SelectorType, config: SelectorConfig):
        if selector_type == SelectorType.RANDOM:
            if not isinstance(config, RandomSelectorConfig):
                raise ValueError("Random selector requires RandomSelectorConfig")
            return RandomSelector(config.homo_g, config.node_mapping, config.primary_type)

        elif selector_type == SelectorType.DEGREE:
            if not isinstance(config, DegreeSelectorConfig):
                raise ValueError("Degree selector requires DegreeSelectorConfig")
            return DegreeBasedSelector(config.homo_g, config.node_mapping, config.primary_type)

        elif selector_type == SelectorType.PAGERANK:
            if not isinstance(config, PageRankSelectorConfig):
                raise ValueError("PageRank selector requires PageRankSelectorConfig")
            return PageRankBasedSelector(
                config.homo_g,
                config.node_mapping,
                config.primary_type,
                alpha=config.alpha
            )

        elif selector_type == SelectorType.LLM:
            if not isinstance(config, LLMSelectorConfig):
                raise ValueError("LLM selector requires LLMSelectorConfig")
            return LLMBasedSelector(
                config.homo_g,
                config.node_mapping,
                config.primary_type,
                config.text_attribute
            )

        elif selector_type == SelectorType.CLUSTER:
            if not isinstance(config, ClusterSelectorConfig):
                raise ValueError("Cluster selector requires ClusterSelectorConfig")
            return ClusterBasedSelector(
                config.homo_g,
                config.node_mapping,
                config.primary_type,
                config.hg
            )