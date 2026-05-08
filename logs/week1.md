# Week 1 — Project Setup & Data Pipeline

## What I Did

Set up the KGround project from scratch. Installed and verified
the full environment: PyTorch 2.5.1 with CUDA 12.1, PyTorch
Geometric, HuggingFace datasets and transformers, bitsandbytes.

Downloaded and explored two datasets:
- Raw WebQSP from Microsoft (4737 QA pairs with SPARQL annotations)
- RoG-WebQSP from HuggingFace (same questions but with pre-extracted
  Freebase subgraphs per question, ~517MB)

Built two dataset classes in src/data/loader.py:
- WebQSPDataset: clean wrapper returning question, topic entity,
  answer entity, local subgraph per example
- KGTripleDataset: builds entity/relation vocabularies across all
  three splits, converts triples to integer indices, caches
  everything to disk

## What the Data Looks Like

Each WebQSP example has a natural language question, a topic entity
(the starting node in the graph, already linked to Freebase), a gold
answer entity, and a local subgraph of. The subgraph
is noisy — multi-hop Freebase neighborhoods pull in loosely connected
entities. The model needs to learn to follow the relevant path and
ignore the noise.

Full vocab across all splits: 1,298,306 entities, 6,094 relations,
~12M train triples, ~7M test triples.

## Problems I Hit

- Windows CP1252 encoding error when inspecting the raw  - WebQSP JSON manually — not part of the actual pipeline
  adding encoding='utf-8' to
- Called .mkdir() on file paths instead of directory paths — pickle
  creates files automatically, only the parent directory needs to exist
- Initially built vocab from train split only — causes OOV at test
  time since test subgraphs contain entities not seen in training.
  Fixed by building vocab across all three splits combined before
  saving to cache

## What I Actually Understood Today

KGQA is not just "query a database". The challenge is that the model
receives a natural language question and a noisy local subgraph and
has to figure out which path through the graph leads to the correct
answer entity — without seeing the gold reasoning chain. That's the
hard part.

Real ML work is mostly data engineering. The model hasn't been written
yet and I already spent two full sessions just getting the data into
a clean, reusable format. Precompute once, cache to disk, reuse forever.

## Next

Read TransE, DistMult, RotatE papers. Start RotatE implementation.