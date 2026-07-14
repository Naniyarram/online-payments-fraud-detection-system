import pandas as pd
import networkx as nx
from typing import Dict, Tuple
from src.utils.logger import get_logger

logger = get_logger(__name__)

def build_network_features(df: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Constructs a Bipartite Customer-Merchant network from transactions.
    Extracts node-level features (PageRank metrics) to model network topology.
    Returns:
        customer_pagerank: Dict mapping cc_num to PageRank score
        merchant_pagerank: Dict mapping merchant to PageRank score
    """
    logger.info("Building Customer-Merchant Graph using NetworkX...")
    
    # Create Graph
    G = nx.Graph()
    
    # We will cast cc_num to string to differentiate nodes from merchants if needed,
    # but cc_num are ints, merchant are strings.
    # Add nodes and edges
    # For speed on large datasets, build edges from zip
    edges = list(zip(df['cc_num'].astype(str), df['merchant']))
    G.add_edges_from(edges)
    
    logger.info(f"Graph constructed with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    logger.info("Computing PageRank scores...")
    # Max iterations capped for fast execution
    pagerank = nx.pagerank(G, max_iter=100)
    
    # Separate customer and merchant PageRank mapping
    customer_pr = {}
    merchant_pr = {}
    
    for node, pr_val in pagerank.items():
        if node.isdigit(): # customer cc_num
            customer_pr[int(node)] = pr_val
        else: # merchant name
            merchant_pr[node] = pr_val
            
    logger.info("PageRank computation complete.")
    return customer_pr, merchant_pr
