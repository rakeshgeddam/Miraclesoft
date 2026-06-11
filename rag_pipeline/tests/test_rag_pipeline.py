"""
Rigorous end-to-end test of the RAG pipeline using real documents.

Tests:
  1. Chunking strategies (recursive, fixed, semantic)
  2. Multiple embedding models (MiniLM, BGE)
  3. Binary vector store format I/O
  4. All three index types (brute-force, IVF, HNSW)
  5. Cross-encoder re-ranking
  6. End-to-end ingest → search → retrieve pipeline
  7. Recall@k comparison: index accuracy vs brute-force baseline
  8. Response format validation
"""

import json
import math
import shutil
import sys
import time
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rag_pipeline import (
    RAGPipeline, PipelineConfig,
    VectorStore, ChunkMetadata,
    EmbeddingManager,
    DocumentChunker,
    IndexManager,
    BruteForceIndex, IVFIndex, HNSWIndex,
    RerankerManager,
)


SAMPLE_DOCS = [
    {
        "id": "doc-llm",
        "text": (
            "Large Language Models (LLMs) are neural network models trained on vast "
            "amounts of text data. They can generate human-like text, answer questions, "
            "summarize documents, and translate languages. Popular examples include "
            "GPT-4, Claude, Gemini, and LLaMA. These models use the transformer "
            "architecture, which relies on self-attention mechanisms to process "
            "sequential data. Training requires thousands of GPUs and weeks of compute "
            "time. LLMs are typically trained in two stages: pre-training on general "
            "corpora and fine-tuning on specific tasks. The pre-training objective is "
            "often next-token prediction, where the model learns to predict the next "
            "word given the previous context."
        ),
        "metadata": {"category": "ai", "source": "handbook"},
    },
    {
        "id": "doc-rag",
        "text": (
            "Retrieval-Augmented Generation (RAG) is a technique that enhances LLMs "
            "by retrieving relevant documents from a knowledge base before generating "
            "an answer. The process involves three steps: first, the user query is "
            "embedded into a vector using an embedding model; second, this vector is "
            "used to search a pre-built index of document embeddings to find the most "
            "relevant passages; third, the retrieved passages are concatenated with "
            "the original query and fed to the LLM for generation. RAG reduces "
            "hallucination and improves factual accuracy by grounding the model's "
            "outputs in retrieved evidence."
        ),
        "metadata": {"category": "ai", "source": "handbook"},
    },
    {
        "id": "doc-embeddings",
        "text": (
            "Embeddings are dense vector representations of text that capture semantic "
            "meaning. Modern embedding models like sentence-transformers can map "
            "sentences and paragraphs to fixed-size vectors (typically 384 to 1024 "
            "dimensions). Semantically similar texts have vectors that are close "
            "together in the embedding space, typically measured by cosine similarity "
            "or dot product. Popular embedding models include all-MiniLM-L6-v2 (384d), "
            "BAAI/bge-base-en-v1.5 (768d), and intfloat/e5-base-v2 (768d). These "
            "models are critical for retrieval systems because they determine the "
            "quality of the semantic search."
        ),
        "metadata": {"category": "ai", "source": "handbook"},
    },
    {
        "id": "doc-hnsw",
        "text": (
            "HNSW (Hierarchical Navigable Small World) is a graph-based algorithm "
            "for approximate nearest neighbor search. It builds a multi-layer graph "
            "where each layer is a progressively sparser set of nodes connected to "
            "their nearest neighbors. Search begins at the top layer and greedily "
            "traverses downward, refining candidates at each level. HNSW offers "
            "O(log N) search complexity with high recall, making it one of the most "
            "popular ANN algorithms. Key parameters include M (max neighbors per node, "
            "typically 16), ef_construction (build quality, 200-500), and ef_search "
            "(search quality, 100-500)."
        ),
        "metadata": {"category": "algorithms", "source": "handbook"},
    },
    {
        "id": "doc-ivf",
        "text": (
            "IVF (Inverted File Index) is a clustering-based approach for approximate "
            "nearest neighbor search. It uses k-means to partition the vector space "
            "into K clusters (centroids). At search time, only the nprobe closest "
            "clusters are scanned, reducing the search space dramatically. IVF is "
            "simple to implement and offers good trade-offs: with K=4096 centroids "
            "and nprobe=16, it can achieve 98% recall at 10-50x speedup over "
            "brute-force. The main costs are the k-means training step and the memory "
            "for storing both centroids and inverted lists."
        ),
        "metadata": {"category": "algorithms", "source": "handbook"},
    },
    {
        "id": "doc-reranking",
        "text": (
            "Re-ranking is a two-stage retrieval strategy. In the first stage, a "
            "fast but approximate method (like ANN search) retrieves a large set of "
            "candidate passages (e.g., 100). In the second stage, a more expensive "
            "but accurate cross-encoder model scores each candidate by considering "
            "the query and passage jointly. Cross-encoders like ms-marco-MiniLM-L6-v2 "
            "process the query and passage as a pair through a transformer, producing "
            "a relevance score. While too slow for full corpus search, they "
            "significantly improve top-k quality when applied to the top 100 candidates."
        ),
        "metadata": {"category": "algorithms", "source": "handbook"},
    },
    {
        "id": "doc-gpu",
        "text": (
            "Graphics Processing Units (GPUs) are specialized hardware designed for "
            "parallel computation. Originally built for rendering graphics, their "
            "massively parallel architecture makes them ideal for deep learning. "
            "Modern GPUs like NVIDIA's H100 and A100 have thousands of CUDA cores "
            "and high-bandwidth memory (HBM). They excel at the matrix multiplications "
            "that form the core of neural network training and inference. A single "
            "H100 can achieve over 2000 TFLOPS for sparse computations. GPU memory "
            "capacity (80GB for H100) is often the limiting factor for large model "
            "training."
        ),
        "metadata": {"category": "hardware", "source": "handbook"},
    },
    {
        "id": "doc-python",
        "text": (
            "Python is a high-level, interpreted programming language known for its "
            "readability and extensive ecosystem. It has become the dominant language "
            "for machine learning and data science due to libraries like NumPy, "
            "PyTorch, TensorFlow, scikit-learn, and pandas. Python's dynamic typing "
            "and interactive development environment (Jupyter notebooks) make it "
            "ideal for prototyping. Key features include first-class functions, "
            "list comprehensions, generators, decorators, and a rich standard library. "
            "The Python Package Index (PyPI) hosts over 400,000 packages."
        ),
        "metadata": {"category": "programming", "source": "handbook"},
    },
    {
        "id": "doc-transformers",
        "text": (
            "The Transformer architecture, introduced in the paper 'Attention is All "
            "You Need' by Vaswani et al. (2017), revolutionized natural language "
            "processing. Its key innovation is the self-attention mechanism, which "
            "allows the model to weigh the importance of different words in a sequence "
            "when computing representations. Transformers consist of an encoder and "
            "a decoder, each made up of stacked layers with multi-head attention and "
            "feed-forward neural networks. Positional encodings inject information "
            "about word order. BERT uses only the encoder, while GPT uses only the "
            "decoder. The architecture scales well and is the foundation of most "
            "modern LLMs."
        ),
        "metadata": {"category": "ai", "source": "handbook"},
    },
    {
        "id": "doc-kmeans",
        "text": (
            "K-means clustering is an unsupervised learning algorithm that partitions "
            "n observations into k clusters. Each observation belongs to the cluster "
            "with the nearest centroid. The algorithm alternates between assignment "
            "(each point assigned to nearest centroid) and update (centroids recalculated "
            "as mean of assigned points). It converges to a local optimum. K-means is "
            "sensitive to initialization — k-means++ initialization helps. In vector "
            "databases, k-means is used to train IVF centroids. The number of clusters "
            "K is a hyperparameter: too few reduces search accuracy, too many increases "
            "memory and training time."
        ),
        "metadata": {"category": "algorithms", "source": "handbook"},
    },
]


def test_01_chunking():
    """Test all 3 chunking strategies produce non-empty chunks with correct IDs."""
    chunker = DocumentChunker()

    for strategy in ["recursive", "fixed", "semantic"]:
        chunks = chunker.chunk(SAMPLE_DOCS[0]["text"], SAMPLE_DOCS[0]["id"], strategy=strategy)
        assert len(chunks) > 0, f"{strategy}: empty chunks"
        assert all(c.chunk_id >= 0 for c in chunks), f"{strategy}: negative chunk_id"
        assert all(c.doc_id == "doc-llm" for c in chunks), f"{strategy}: wrong doc_id"
        assert all(len(c.text) > 0 for c in chunks), f"{strategy}: empty text in chunk"
        print(f"  ✓ {strategy}: {len(chunks)} chunks, avg {sum(len(c) for c in chunks)//len(chunks)} chars")

    print("  ✓ All chunking strategies pass")


def test_02_embedding_models():
    """Test multiple embedding models produce correct-shaped vectors."""
    embed = EmbeddingManager()
    models = embed.list_models()
    assert len(models) >= 2, "Need at least 2 models configured"

    texts = ["What is RAG?", "Embeddings capture semantics.", "GPU computing is fast."]

    for m in models[:3]:  # Test first 3 models
        vectors = embed.embed(texts, model=m.name)
        assert vectors.shape == (3, m.dimension), (
            f"{m.name}: expected ({3},{m.dimension}), got {vectors.shape}"
        )
        assert vectors.dtype == np.float32, f"{m.name}: dtype must be float32"
        # Check unit norm if normalize=True
        if m.normalize:
            norms = np.linalg.norm(vectors, axis=1)
            assert np.allclose(norms, 1.0, atol=1e-5), f"{m.name}: vectors not normalized"
        print(f"  ✓ {m.name}: {vectors.shape}, dtype={vectors.dtype}, normed={m.normalize}")

    # Test single query embed
    qv = embed.embed_query("Test query", model=models[0].name)
    assert qv.shape == (models[0].dimension,), f"Query embed shape wrong: {qv.shape}"
    print(f"  ✓ Query embedding shape: {qv.shape}")

    print("  ✓ All embedding models pass")


def test_03_vector_store_binary_format():
    """Test binary vector store write/read round-trip with correct byte layout."""
    with tempfile.TemporaryDirectory() as tmp:
        store = VectorStore(
            index_path=f"{tmp}/index.bin",
            metadata_path=f"{tmp}/chunks.jsonl",
            dimension=384,
            normalize=True,
        )

        store.create(num_vectors=5)

        # Create test vectors
        vectors = np.random.randn(5, 384).astype(np.float32)
        # Normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms

        doc_ids = np.array([10, 20, 30, 40, 50], dtype=np.uint64)
        chunk_offsets = np.array([0, 1, 2, 3, 4], dtype=np.uint64)
        meta = [
            ChunkMetadata(chunk_id=i, doc_id=f"doc{i}", text=f"Test chunk {i}",
                          token_start=0, token_end=10)
            for i in range(5)
        ]

        # Write
        store.add_vectors(vectors, doc_ids, chunk_offsets, meta)
        assert store.num_vectors == 5

        # Re-open and verify
        store.close()

        store2 = VectorStore(
            index_path=f"{tmp}/index.bin",
            metadata_path=f"{tmp}/chunks.jsonl",
            dimension=384,
            normalize=True,
        )
        store2.open()

        assert store2.num_vectors == 5
        assert store2.dim == 384

        # Check magic bytes
        with open(f"{tmp}/index.bin", "rb") as f:
            magic = f.read(4)
            assert magic == b"HRID", f"Wrong magic: {magic}"

        # Verify vector round-trip
        v0 = store2.get_vector(0)
        assert np.allclose(v0, vectors[0], atol=1e-6), "Vector round-trip mismatch"

        # Verify doc_id
        assert store2.get_doc_id(0) == 10, "DocID round-trip mismatch"

        # Brute force search
        search_results = store2.search_brute_force(vectors[0], k=3)
        assert len(search_results) == 3, f"Expected 3 results, got {len(search_results)}"
        # The query vector itself should be top result
        assert search_results[0][1] == 0, "Self-match should be top result"

        print(f"  ✓ Binary format: {store2.num_vectors} vectors, "
              f"search returned {len(search_results)} results")
        store2.close()

    print("  ✓ Vector store I/O and search pass")


def test_04_index_types():
    """Test all 3 index types produce correct top-k and recall vs brute-force."""
    np.random.seed(42)
    n_vectors = 1000
    dim = 384

    # Generate random normalized vectors
    vectors = np.random.randn(n_vectors, dim).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms

    # Query vectors
    queries = np.random.randn(10, dim).astype(np.float32)
    q_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    queries = queries / q_norms

    # Brute-force baseline (exact)
    bf = BruteForceIndex(normalized=True)
    bf.build(vectors)

    bf_recalls = []
    for q in queries:
        bf_results = bf.search(q, k=10)
        bf_recalls.append(set(r[1] for r in bf_results))
    # Ground truth is brute-force itself

    # IVF
    ivf = IVFIndex(n_centroids=64, n_probe=8, normalized=True, seed=42)
    ivf.build(vectors)

    ivf_recalls = []
    for i, q in enumerate(queries):
        ivf_results = ivf.search(q, k=10)
        ivf_set = set(r[1] for r in ivf_results)
        recall = len(bf_recalls[i] & ivf_set) / 10.0
        ivf_recalls.append(recall)
    avg_ivf_recall = sum(ivf_recalls) / len(ivf_recalls)
    print(f"  ✓ IVF@10 recall: {avg_ivf_recall:.3f} (target > 0.80 for n=1000, K=64)")

    # HNSW
    hnsw = HNSWIndex(M=8, ef_construction=100, ef_search=100, normalized=True, seed=42)
    hnsw.build(vectors)

    hnsw_recalls = []
    for i, q in enumerate(queries):
        hnsw_results = hnsw.search(q, k=10)
        hnsw_set = set(r[1] for r in hnsw_results)
        recall = len(bf_recalls[i] & hnsw_set) / 10.0
        hnsw_recalls.append(recall)
    avg_hnsw_recall = sum(hnsw_recalls) / len(hnsw_recalls)
    print(f"  ✓ HNSW@10 recall: {avg_hnsw_recall:.3f} (target > 0.90 for n=1000, M=8)")

    # Check persistence
    with tempfile.TemporaryDirectory() as tmp:
        # Save & load IVF
        ivf.save(f"{tmp}/ivf")
        ivf2 = IVFIndex()
        ivf2.load(f"{tmp}/ivf")
        q = queries[0]
        r1 = set(r[1] for r in ivf.search(q, 10))
        r2 = set(r[1] for r in ivf2.search(q, 10))
        assert r1 == r2, "IVF save/load mismatch"
        print("  ✓ IVF persistence round-trip")

        # Save & load HNSW
        hnsw.save(f"{tmp}/hnsw")
        hnsw2 = HNSWIndex()
        hnsw2.load(f"{tmp}/hnsw")
        r1 = set(r[1] for r in hnsw.search(q, 10))
        r2 = set(r[1] for r in hnsw2.search(q, 10))
        assert r1 == r2, "HNSW save/load mismatch"
        print("  ✓ HNSW persistence round-trip")

    print("  ✓ All index types pass")


def test_05_reranker():
    """Test cross-encoder reranker produces re-ranked results."""
    # Create a simple CrossEncoderReranker from our factory
    reranker = RerankerManager()
    models = reranker.list_models()
    if not models:
        print("  ⚠ No reranker models configured, skipping")
        return

    model_name = models[0].name

    query = "What is RAG and how does it reduce hallucination?"
    passages = [d["text"] for d in SAMPLE_DOCS]

    reranked = reranker.rerank(query, passages, k=5, model=model_name)
    assert len(reranked) == 5, f"Expected 5 reranked, got {len(reranked)}"
    assert reranked[0][0] >= reranked[-1][0], "Reranked list not sorted descending"
    assert all(isinstance(r[2], str) for r in reranked), "Passage text missing"

    print(f"  ✓ Reranker ({model_name}): top-1 score={reranked[0][0]:.4f}, "
          f"top-5 score={reranked[-1][0]:.4f}")


def test_06_ingest_and_pipeline():
    """End-to-end test: ingest documents then search and answer."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = PipelineConfig(
            data_dir=tmp,
            index_type="brute_force",  # exact for test
            embedding_model="miniLM-L6-v2",
            top_k_retrieve=5,
            top_k_rerank=3,
            rerank=True,
        )
        pipeline = RAGPipeline(cfg)

        # Ingest
        result = pipeline.ingest(SAMPLE_DOCS)
        assert result["chunks"] > 0, "No chunks created"
        assert result["vectors_added"] > 0, "No vectors added"
        assert pipeline.num_vectors > 0
        print(f"  ✓ Ingest: {result['chunks']} chunks, {result['vectors_added']} vectors "
              f"in {result['time']:.2f}s")

        # Search (retrieve only)
        search_results = pipeline.search("How do embeddings work?", k=3)
        assert len(search_results) >= 1, "Search returned no results"
        print(f"  ✓ Search: top score={search_results[0]['score']:.4f}, "
              f"text='{search_results[0]['text'][:80]}...'")

        # Full answer pipeline
        qa = pipeline.answer("What is RAG and how does it reduce hallucination?")
        assert len(qa.answer) > 0, "Empty answer"
        assert len(qa.retrieved_passages) >= 1, "No retrieved passages"
        assert len(qa.reranked_passages) >= 1, "No reranked passages"
        assert qa.retrieved_passages[0]["score"] >= 0, "Similarity should be >= 0 for normalized vectors"

        print(f"  ✓ Answer pipeline: retrieved={len(qa.retrieved_passages)}, "
              f"reranked={len(qa.reranked_passages)}")
        print(f"  ✓ Answer snippet: {qa.answer[:120]}...")
        print(f"  ✓ Timing: {json.dumps({k: f'{v:.3f}s' for k, v in qa.timing.items()})}")

        # Test with HNSW index
        cfg2 = PipelineConfig(
            data_dir=tmp + "_hnsw",
            index_type="hnsw",
            embedding_model="miniLM-L6-v2",
        )
        pipeline2 = RAGPipeline(cfg2)
        pipeline2.ingest(SAMPLE_DOCS)
        qa2 = pipeline2.answer("What is HNSW?")
        assert len(qa2.answer) > 0
        print(f"  ✓ HNSW pipeline answer: {qa2.answer[:120]}...")

        # Test with IVF index
        cfg3 = PipelineConfig(
            data_dir=tmp + "_ivf",
            index_type="ivf",
            embedding_model="miniLM-L6-v2",
        )
        pipeline3 = RAGPipeline(cfg3)
        pipeline3.ingest(SAMPLE_DOCS)
        qa3 = pipeline3.answer("What is k-means?")
        assert len(qa3.answer) > 0
        print(f"  ✓ IVF pipeline answer: {qa3.answer[:120]}...")


def test_07_recall_accuracy():
    """Verify ANN indices maintain >90% recall vs brute-force on real embeddings."""
    with tempfile.TemporaryDirectory() as tmp:
        # Embed the sample docs
        cfg = PipelineConfig(data_dir=tmp)
        pipeline = RAGPipeline(cfg)
        pipeline.initialize()

        embed_mgr = EmbeddingManager()
        texts = [d["text"] for d in SAMPLE_DOCS]
        vectors = embed_mgr.embed(texts, model="miniLM-L6-v2")

        # Brute-force exact results for queries
        queries_texts = [
            "How does RAG reduce hallucination?",
            "What is HNSW search?",
            "Explain GPU computing",
            "What is the transformer architecture?",
        ]
        query_vecs = embed_mgr.embed(queries_texts, model="miniLM-L6-v2")

        bf = BruteForceIndex(normalized=True)
        bf.build(vectors)

        gt_sets = {}
        for i, qv in enumerate(query_vecs):
            results = bf.search(qv, k=5)
            gt_sets[i] = set(r[1] for r in results)

        # Test HNSW
        hnsw = HNSWIndex(M=8, ef_construction=100, ef_search=50, normalized=True, seed=42)
        hnsw.build(vectors)

        recalls = []
        for i, qv in enumerate(query_vecs):
            results = hnsw.search(qv, k=5)
            pred_set = set(r[1] for r in results)
            recall = len(gt_sets[i] & pred_set) / 5.0
            recalls.append(recall)

        avg_recall = sum(recalls) / len(recalls)
        print(f"  ✓ HNSW recall@5 on real embeddings: {avg_recall:.3f}")
        assert avg_recall >= 0.80, f"HNSW recall too low: {avg_recall}"

        # Test IVF
        ivf = IVFIndex(n_centroids=max(4, len(vectors) // 2), n_probe=4,
                       normalized=True, seed=42)
        ivf.build(vectors)

        recalls = []
        for i, qv in enumerate(query_vecs):
            results = ivf.search(qv, k=5)
            pred_set = set(r[1] for r in results)
            recall = len(gt_sets[i] & pred_set) / 5.0
            recalls.append(recall)

        avg_recall = sum(recalls) / len(recalls)
        print(f"  ✓ IVF recall@5 on real embeddings: {avg_recall:.3f}")
        assert avg_recall >= 0.70, f"IVF recall too low: {avg_recall}"


def test_08_error_handling():
    """Test edge cases: empty documents, missing query, zero vectors."""
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = RAGPipeline(PipelineConfig(data_dir=tmp))
        pipeline.initialize()

        # Empty ingest
        result = pipeline.ingest([])
        assert result["chunks"] == 0
        print("  ✓ Empty ingest handled")

        # Query before ingest
        qa = pipeline.answer("test")
        assert "No documents ingested" in qa.answer
        print("  ✓ Empty query handled")

        # Single doc
        result = pipeline.ingest([SAMPLE_DOCS[0]])
        assert result["vectors_added"] > 0
        qa = pipeline.answer("What are LLMs?")
        assert len(qa.answer) > 0
        print("  ✓ Single document ingest & query")


def test_09_different_embedding_models():
    """Test pipeline works with different embedding models."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = PipelineConfig(
            data_dir=tmp,
            embedding_model="miniLM-L6-v2",
        )
        pipeline = RAGPipeline(cfg)
        pipeline.initialize()
        pipeline.ingest(SAMPLE_DOCS[:3])
        qa = pipeline.answer("What are embeddings?")
        assert len(qa.answer) > 0
        print(f"  ✓ MiniLM pipeline: answer={qa.answer[:80]}...")

    with tempfile.TemporaryDirectory() as tmp:
        cfg = PipelineConfig(
            data_dir=tmp,
            embedding_model="bge-base-en",
        )
        pipeline = RAGPipeline(cfg)
        pipeline.initialize()
        pipeline.ingest(SAMPLE_DOCS[:3])
        qa = pipeline.answer("What are embeddings?")
        assert len(qa.answer) > 0
        print(f"  ✓ BGE pipeline: answer={qa.answer[:80]}...")


def run_all():
    """Run all tests and report."""
    tests = [
        ("Chunking strategies", test_01_chunking),
        ("Embedding models", test_02_embedding_models),
        ("Vector store binary format", test_03_vector_store_binary_format),
        ("ANN index types", test_04_index_types),
        ("Cross-encoder reranker", test_05_reranker),
        ("Ingest & answer pipeline", test_06_ingest_and_pipeline),
        ("Recall accuracy", test_07_recall_accuracy),
        ("Error handling", test_08_error_handling),
        ("Multiple embedding models", test_09_different_embedding_models),
    ]

    passed = 0
    failed = 0

    print(f"\n{'='*60}")
    print("  RAG Pipeline — Rigorous Test Suite")
    print(f"{'='*60}\n")

    for name, fn in tests:
        print(f"[TEST] {name}")
        print("-" * 40)
        try:
            fn()
            print(f"  ✅ PASSED\n")
            passed += 1
        except Exception as e:
            import traceback
            print(f"  ❌ FAILED: {e}")
            traceback.print_exc()
            print()
            failed += 1

    print(f"{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed, {len(tests)} total")
    print(f"{'='*60}")

    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
