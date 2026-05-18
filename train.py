"""
dataset.py

Handles tokenisation using spaCy and dataset creation for Multi30k.
Vocabularies are built with torchtext.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
import spacy
from torchtext.vocab import build_vocab_from_iterator

SPECIAL_TOKENS = ["<unk>", "<pad>", "<sos>", "<eos>"]
UNK_IDX = 0
PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

def fetch_spacy_models():
    try:
        de_model = spacy.load("de_core_news_sm")
    except OSError:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "de_core_news_sm"], check=True)
        de_model = spacy.load("de_core_news_sm")

    try:
        en_model = spacy.load("en_core_web_sm")
    except OSError:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
        en_model = spacy.load("en_core_web_sm")

    return de_model, en_model

def extract_german_tokens(text_input, nlp_pipeline):
    return [token.text.lower() for token in nlp_pipeline(text_input)]

def extract_english_tokens(text_input, nlp_pipeline):
    return [token.text.lower() for token in nlp_pipeline(text_input)]

class Multi30kDataset(Dataset):
    def __init__(self, split='train', src_vocab=None, tgt_vocab=None):
        self.data_split = split
        self.german_nlp, self.english_nlp = fetch_spacy_models()
        
        # FIXED: Removed trust_remote_code=True to resolve the Hugging Face deprecation warning
        hf_dataset = load_dataset("bentrevett/multi30k")
        self.raw_corpus = hf_dataset[split]
        
        if src_vocab is None or tgt_vocab is None:
            assert split == "train", "Vocabs must be provided for non-train splits."
            self.src_vocab, self.tgt_vocab = self.build_vocab()
        else:
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab
            
        self.processed_samples = self.process_data()

    def build_vocab(self):
        def yield_german_tokens():
            for item in self.raw_corpus:
                yield extract_german_tokens(item["de"], self.german_nlp)
                
        def yield_english_tokens():
            for item in self.raw_corpus:
                yield extract_english_tokens(item["en"], self.english_nlp)
                
        source_v = build_vocab_from_iterator(yield_german_tokens(), specials=SPECIAL_TOKENS, special_first=True)
        source_v.set_default_index(UNK_IDX)
        
        target_v = build_vocab_from_iterator(yield_english_tokens(), specials=SPECIAL_TOKENS, special_first=True)
        target_v.set_default_index(UNK_IDX)
        
        return source_v, target_v

    def process_data(self):
        tensor_pairs = []
        for item in self.raw_corpus:
            de_tokens = extract_german_tokens(item["de"], self.german_nlp)
            en_tokens = extract_english_tokens(item["en"], self.english_nlp)
            
            de_indices = [SOS_IDX] + self.src_vocab(de_tokens) + [EOS_IDX]
            en_indices = [SOS_IDX] + self.tgt_vocab(en_tokens) + [EOS_IDX]
            
            tensor_pairs.append((
                torch.tensor(de_indices, dtype=torch.long),
                torch.tensor(en_indices, dtype=torch.long)
            ))
        return tensor_pairs

    def __len__(self):
        return len(self.processed_samples)

    def __getitem__(self, index):
        return self.processed_samples[index]

def pad_collate_function(batch_items, padding_value=PAD_IDX):
    source_sequences, target_sequences = zip(*batch_items)
    padded_source = pad_sequence(source_sequences, batch_first=True, padding_value=padding_value)
    padded_target = pad_sequence(target_sequences, batch_first=True, padding_value=padding_value)
    return padded_source, padded_target

def build_dataloaders(batch_sz=128):
    train_ds = Multi30kDataset(split="train")
    src_v = train_ds.src_vocab
    tgt_v = train_ds.tgt_vocab
    
    val_ds = Multi30kDataset(split="validation", src_vocab=src_v, tgt_vocab=tgt_v)
    test_ds = Multi30kDataset(split="test", src_vocab=src_v, tgt_vocab=tgt_v)
    
    collate_fn_bound = lambda b: pad_collate_function(b, padding_value=PAD_IDX)
    
    loader_train = DataLoader(train_ds, batch_size=batch_sz, shuffle=True, collate_fn=collate_fn_bound)
    loader_val = DataLoader(val_ds, batch_size=batch_sz, shuffle=False, collate_fn=collate_fn_bound)
    loader_test = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn_bound)
    
    return loader_train, loader_val, loader_test, src_v, tgt_v