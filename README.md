BIS-Hackathon - Sigma-Squad
# BIS Standards Recommendation Engine

**Overview**

This project recommends relevant **BIS standards** based on a product description using a **RAG (Retrieval-Augmented Generation)** approach.

Input → Product description
Output → Top 3–5 BIS standards with reasoning

**How it works**

```id="f0x8jr"
Query → Retrieve → Rerank → Generate → Output
```

* **Retrieve**: Get relevant standards (Person A - SARA Y)
* **Rerank**: Select best ones (Person B - SHREE RATHINA KUMAR)
* **Generate**: Produce final answer (Person C - ANUSHRI RAJKUMAR)

**Project Structure**

```id="h3o2r7"
project/
│
├── src/
│   ├── retriever.py
│   ├── reranker.py
│   ├── generator.py
│   ├── pipeline.py
│
├── inference.py
├── eval_script.py
├── public_test_set.json
├── requirements.txt
```

---

## How to run

### 1. Install dependencies

```id="jj6wtt"
pip install -r requirements.txt
```

### 2. Run the system

```id="nix0mh"
python inference.py --input public_test_set.json --output output.json
```

### 3. Evaluate

```id="z6jv2f"
python eval_script.py --results output.json
```

---

## Output format

```id="yk8k0m"
[
  {
    "id": "Q1",
    "retrieved_standards": ["IS 1786", "IS 456"],
    "latency_seconds": 1.2
  }
]
```

---

## Key points

* Returns **top 3–5 standards**
* **No hallucination** (only from retrieved data)
* **Fast response (<5 seconds)**
* Works with provided evaluation script

---

## Team roles

* Person A - SARA Y → Retrieval
* Person B - SHREE RATHINA KUMAR P → Reranking
* Person C - ANUSHRI RAJKUMAR → Generation & Inference

---

## Summary

A simple and efficient system to help users quickly find the correct BIS standards.

---
