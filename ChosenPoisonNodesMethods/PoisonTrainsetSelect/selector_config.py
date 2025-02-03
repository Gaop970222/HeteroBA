from dataclasses import dataclass
from typing import Optional, Any

@dataclass
class SelectorConfig:
    homo_g: Any
    node_mapping: dict
    primary_type: str

@dataclass
class RandomSelectorConfig(SelectorConfig):
    pass 
    
@dataclass
class DegreeSelectorConfig(SelectorConfig):
    pass 

@dataclass
class PageRankSelectorConfig(SelectorConfig):
    alpha: float = 0.85

@dataclass
class LLMSelectorConfig(SelectorConfig):
    text_attribute: str

@dataclass
class ClusterSelectorConfig(SelectorConfig):
    hg: Any
    # 可以添加其他聚类特定的参数
    n_clusters: int = 5