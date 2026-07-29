import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Dict, Any
from src.utils.logger import get_logger

logger = get_logger(__name__)

PAST_CASES = [
    {
        "id": "case_001",
        "text": "Cardholder flagged transaction for high-value electronics purchase in a high-risk state. Pattern matches 'mule redirection' where shipping address differs from cardholder home state.",
        "category": "shopping_net", "is_fraud": 1
    },
    {
        "id": "case_002",
        "text": "Multiple microtransactions under $5 executed within 1 minute, followed by a large $1000 cash withdrawal. Typical card-testing behaviour prior to liquidation.",
        "category": "gas_transport", "is_fraud": 1
    },
    {
        "id": "case_003",
        "text": "Transaction declined due to sudden geographic jump over 500 km from previous transaction location under 30 minutes. Pattern matches physical card cloning.",
        "category": "grocery_pos", "is_fraud": 1
    },
    {
        "id": "case_004",
        "text": "High volume of entertainment and restaurant transactions at late night hours. Customer disputes charges. Investigated as stolen physical credit card.",
        "category": "entertainment", "is_fraud": 1
    },
    {
        "id": "case_005",
        "text": "Cardholder velocity spike: 8 luxury fashion orders within 2 hours totaling $4,500 across 3 different online merchants in NY.",
        "category": "shopping_net", "is_fraud": 1
    },
    {
        "id": "case_006",
        "text": "Cross-border travel purchase originating from suspicious IP subnet while customer physical location registered in California.",
        "category": "travel", "is_fraud": 1
    },
    {
        "id": "case_007",
        "text": "BIN attack sequence detected with sequential credit card trial attempts at automated fuel pump terminal.",
        "category": "gas_transport", "is_fraud": 1
    },
    {
        "id": "case_008",
        "text": "Account Takeover (ATO): Password reset followed immediately by high-value gift card purchases at online retail store.",
        "category": "shopping_net", "is_fraud": 1
    },
    {
        "id": "case_009",
        "text": "Repeated grocery terminal transactions with abnormal spending 5x above cardholder's 24h baseline average.",
        "category": "grocery_pos", "is_fraud": 1
    },
    {
        "id": "case_010",
        "text": "Legitimate cardholder purchasing routine monthly groceries at local supermarket during normal daytime hours.",
        "category": "grocery_pos", "is_fraud": 0
    },
    {
        "id": "case_011",
        "text": "Approved recurring digital subscription charge for streaming service matching multi-year history.",
        "category": "entertainment", "is_fraud": 0
    },
    {
        "id": "case_012",
        "text": "Standard gas station refueling charge within cardholder home zip code matching regular commuting behavior.",
        "category": "gas_transport", "is_fraud": 0
    }
]

class HybridRetriever:
    def __init__(self, db_path: str = "./data/chroma_db"):
        logger.info("Initializing Hybrid Vector-Sparse Retriever (Lazy Embedding Mode)...")
        self.db_path = db_path
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        self.emb_fn = None
        self.collection = None
        
        # Initialize BM25 Index immediately (lightweight memory footprint)
        self.corpus = PAST_CASES
        self.tokenized_corpus = [c["text"].lower().split(" ") for c in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _get_collection(self):
        """Lazy loader for SentenceTransformer embedding function & ChromaDB collection."""
        if self.collection is None:
            logger.info("Lazy loading SentenceTransformer embedding function ('all-MiniLM-L6-v2')...")
            self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            self.collection = self.chroma_client.get_or_create_collection(
                name="historical_cases",
                embedding_function=self.emb_fn
            )
            if self.collection.count() == 0:
                logger.info("Populating vector store with historical cases...")
                self.collection.add(
                    documents=[c["text"] for c in PAST_CASES],
                    metadatas=[{"category": c["category"], "is_fraud": c["is_fraud"]} for c in PAST_CASES],
                    ids=[c["id"] for c in PAST_CASES]
                )
        return self.collection

    def retrieve(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Executes hybrid search (ChromaDB + BM25) and applies Reciprocal Rank Fusion (RRF).
        """
        logger.info(f"Retrieving historical cases for query: '{query}'")
        
        # 1. Dense (ChromaDB - Lazy Loaded)
        collection = self._get_collection()
        dense_results = collection.query(
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
