import pickle
from pathlib import Path
from datasets import load_dataset
import torch
from torch.utils.data import Dataset

CACHE_DIR = Path('data/cache')
VOCAB_CACHE = CACHE_DIR / 'kg_vocab.pkl'


class WebQSPDataset(Dataset):
    """
    Clean wrapper around RoG-WebQSP.
    Returns question, topic entity, answer entity, and local subgraph.
    """

    def __init__(self, split: str = 'train'):
        assert split in ('train', 'validation', 'test')
        raw = load_dataset('rmanluo/RoG-webqsp')
        self.data = raw[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        return {
            'id':       ex['id'],
            'question': ex['question'],
            'q_entity': ex['q_entity'],
            'a_entity': ex['a_entity'],
            'graph':    ex['graph'],
            'choices':  ex['choices'],
        }


class KGTripleDataset(Dataset):
    """
    Builds entity and relation vocabularies across ALL splits
    so no entity is out-of-vocabulary at test time.
    Caches vocab to disk so it only builds once.
    Loads triples for the requested split only.
    """

    def __init__(self, split: str = 'train'):
        assert split in ('train', 'validation', 'test')
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        TRIPLE_DIR = CACHE_DIR / f'triples_{split}.pkl'

        # load the raw dataset (instant, already cached by HuggingFace)
        raw = load_dataset('rmanluo/RoG-webqsp')

        if VOCAB_CACHE.exists():
            # vocab already built before, load from disk
            print("[KGTripleDataset] Loading vocab from cache...")
            with open(VOCAB_CACHE, 'rb') as f:
                vocab = pickle.load(f)
            self.entity2id   = vocab['entity2id']
            self.relation2id = vocab['relation2id']
            self.id2entity   = vocab['id2entity']
            self.id2relation = vocab['id2relation']

        else:
            # first run: build vocab by iterating all three splits
            print("[KGTripleDataset] Building vocab from scratch (one time only)...")
            entity_set   = set()
            relation_set = set()

            for split_name in ('train', 'validation', 'test'):
                for ex in raw[split_name]:
                    for triple in ex['graph']:
                        s, r, o = triple
                        entity_set.add(s)
                        entity_set.add(o)
                        relation_set.add(r)

            # sort for determinism: same run always gives same IDs
            self.entity2id   = {e: i for i, e in enumerate(sorted(entity_set))}
            self.relation2id = {r: i for i, r in enumerate(sorted(relation_set))}
            self.id2entity   = {i: e for e, i in self.entity2id.items()}
            self.id2relation = {i: r for r, i in self.relation2id.items()}

            # save all four dicts together as one object
            vocab = {
                'entity2id':   self.entity2id,
                'relation2id': self.relation2id,
                'id2entity':   self.id2entity,
                'id2relation': self.id2relation,
            }
            with open(VOCAB_CACHE, 'wb') as f:
                pickle.dump(vocab, f)
            print(f"[KGTripleDataset] Vocab saved to {VOCAB_CACHE}")


        if TRIPLE_DIR.exists():
            # triples already cached from a previous run, load from disk
            print("[KGTripleDataset] Loading triples from cache...")
            with open(TRIPLE_DIR, 'rb') as f:
                self.triples = pickle.load(f) 
        
        else: 
            # first run: convert triples to integer IDs and cache to disk for next time
            print(f"[KGTripleDataset] Converting triples to integer IDs")
            for split_name in ('train', 'validation', 'test'):
                triples = []
                for ex in raw[split_name]:
                    for triple in ex['graph']:
                        s, r, o = triple
                        # convert strings to integer indices
                        triples.append((
                            self.entity2id[s],
                            self.relation2id[r],
                            self.entity2id[o],
                        ))

                with open(CACHE_DIR / f'triples_{split_name}.pkl', 'wb') as f:
                    pickle.dump(triples, f)
                print(f"[KGTripleDataset] Triples for split={split_name} saved to {CACHE_DIR / f'triples_{split_name}.pkl'}")
            
            with open(TRIPLE_DIR, 'rb') as f:
                self.triples = pickle.load(f)


        print(f"[KGTripleDataset] Vocab: {len(self.entity2id)} entities, {len(self.relation2id)} relations")
        print(f"[KGTripleDataset] Split={split}: {len(self.triples)} triples")

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        s, r, o = self.triples[idx]
        return {
            'subject' :     torch.tensor(s, dtype=torch.long),
            'relation':     torch.tensor(r, dtype=torch.long),
            'object'  :     torch.tensor(o, dtype=torch.long),
        }


if __name__ == '__main__':
    print("=== WebQSPDataset ===")
    for split in ('train', 'validation', 'test'):
        ds = WebQSPDataset(split=split)
        print(f"  {split}: {len(ds)} examples")
    ex = WebQSPDataset(split='train')[0]
    print("\nSample example from WebQSPDataset train split:\n")
    print(f"  Sample question: {ex['question']}")
    print(f"  Topic entity:    {ex['q_entity']}")
    print(f"  Answer entity:   {ex['a_entity']}")
    print(f"  Graph triples:   {len(ex['graph'])}")
    print(f"  Choices:         {ex['choices']}")

    print("\n=== KGTripleDataset (first run, will take ~5 min) ===")
    kg = KGTripleDataset(split='train')
    sample = kg[0]
    print(f"  Sample: {kg.id2entity[sample['subject'].item()]} "
          f"--{kg.id2relation[sample['relation'].item()]}--> "
          f"{kg.id2entity[sample['object'].item()]}")

    print("\n=== KGTripleDataset (second run, faster) ===")
    kg_test = KGTripleDataset(split='test')
    print(f"  Test triples: {len(kg_test)}")