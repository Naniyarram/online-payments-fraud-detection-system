import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Dict, Any
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Sample past fraud database templates to populate the database
PAST_CASES = [
    {
        "id": "case_001",
        "text": "Cardholder flagged transaction for electronics purchase in high-risk state. Pattern matches 'mule redirection' where shipping address differs from cardholder home state.",
        "category": "shopping_net", "is_fraud": 1
    },
    {
        "id": "case_002",
        "text": "Multiple microtransactions (under $5) executed within 1 minute, followed by a large $1000 cash withdraw. Typical card-testing behaviour prior to liquidation.",
        "category": "gas_transport", "is_fraud": 1
    },
    {
        "id": "case_003",
        "text": "Transaction declined due to sudden geographic jump (over 500 km) from previous transaction location under 30 minutes. Pattern matches card cloning.",
        "category": "grocery_pos", "is_fraud": 1
    },
    {
        "id": "case_004",
        "text": "High volume of entertainment and restaurant transactions at late night hours. Customer disputes charges. Investigated as stolen physical credit card.",
        "category": "entertainment", "is_fraud": 1
    }
]

class HybridRetriever:
    def __init__(self, db_path: str = "./data/chroma_db"):
        logger.info("Initializing Hybrid Vector-Sparse Retriever...")
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        
        # Use sentence-transformers embedding function
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Setup Collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="historical_cases",
            embedding_function=self.emb_fn
        )
        
        # Populate DB if empty
        if self.collection.count() == 0:
            logger.info("Populating vector store with historical cases...")
            self.collection.add(
                documents=[c["text"] for c in PAST_CASES],
                metadatas=[{"category": c["category"], "is_fraud": c["is_fraud"]} for c in PAST_CASES],
                ids=[c["id"] for c in PAST_CASES]
            )
            
        # Initialize BM25 Index
        self.corpus = PAST_CASES
        self.tokenized_corpus = [c["text"].lower().split(" ") for c in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Executes hybrid search (ChromaDB + BM25) and applies Reciprocal Rank Fusion (RRF).
        """
        logger.info(f"Retrieving historical cases for query: '{query}'")
        
        # 1. Dense (ChromaDB)
        dense_results = self.collection.query(
            query_texts=[query],
            n_results=len(self.corpus)
        )
        
        dense_ranking = []
        if dense_results and dense_results["ids"]:
            dense_ranking = dense_results["ids"][0]
            
        # 2. Sparse (BM25)
        tokenized_query = query.lower().split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        # Sort indices by BM25 score
        sparse_indices = np.argsort(bm25_scores)[::-1]
        sparse_ranking = [self.corpus[idx]["id"] for idx in sparse_indices]
        
        # 3. Reciprocal Rank Fusion (RRF)
        # RRF formula: RRF_score(d) = sum_{m in models} 1 / (k + rank_m(d))
        rrf_k = 60
        rrf_scores = {}
        
        # We index positions (0-based) so rank is index + 1
        for idx, doc_id in enumerate(dense_ranking):
            rank = idx + 1
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank))
            
        for idx, doc_id in enumerate(sparse_ranking):
            rank = idx + 1
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank))
            
        # Sort doc IDs by RRF score descending
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        top_doc_ids = [doc_id for doc_id, score in sorted_docs[:top_k]]
        
        # Retrieve original document contents
        retrieved_cases = []
        for doc_id in top_doc_ids:
            case = next((c for c in self.corpus if c["id"] == doc_id), None)
            if case:
                retrieved_cases.append(case)
                
        logger.info(f"Retrieved {len(retrieved_cases)} cases via Hybrid RRF.")
        return retrieved_cases
