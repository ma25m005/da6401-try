"""
train.py

Training loops, evaluations, Label smoothing loss and greedy generation algorithms.
"""

import os
import math
from collections import Counter
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb
from model import Transformer, make_src_mask, make_tgt_mask
from lr_scheduler import NoamScheduler

PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

class LabelSmoothingLoss(nn.Module):
    """Smoothes cross entropy target to prevent parameter overconfidence."""
    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.num_classes = vocab_size
        self.padding_index = pad_idx
        self.smooth_factor = smoothing
        self.true_prob = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probabilities = torch.log_softmax(logits, dim=-1)
        
        with torch.no_grad():
            smoothed_targets = torch.full_like(
                log_probabilities, self.smooth_factor / (self.num_classes - 2)
            )
            smoothed_targets.scatter_(1, target.unsqueeze(1), self.true_prob)
            smoothed_targets[:, self.padding_index] = 0.0
            
            valid_mask = (target != self.padding_index)
            smoothed_targets[~valid_mask] = 0.0
            
        cross_ent_loss = -(smoothed_targets * log_probabilities).sum()
        valid_tokens_count = valid_mask.sum().float()
        return cross_ent_loss / max(valid_tokens_count, 1.0)

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """Carries out one training or evaluation epoch."""
    model.train() if is_train else model.eval()
    accumulated_loss = 0.0
    accumulated_tokens = 0
    
    grad_context = torch.enable_grad() if is_train else torch.no_grad()
    
    with grad_context:
        for source_batch, target_batch in data_iter:
            source_batch = source_batch.to(device)
            target_batch = target_batch.to(device)
            
            # Form sequences matching teacher forcing paradigm
            decoder_in = target_batch[:, :-1]
            decoder_out = target_batch[:, 1:]
            
            mask_src = make_src_mask(source_batch, pad_idx=PAD_IDX).to(device)
            mask_tgt = make_tgt_mask(decoder_in, pad_idx=PAD_IDX).to(device)
            
            predictions = model(source_batch, decoder_in, mask_src, mask_tgt)
            predictions_flat = predictions.reshape(-1, predictions.size(-1))
            targets_flat = decoder_out.reshape(-1)
            
            batch_loss = loss_fn(predictions_flat, targets_flat)
            
            if is_train:
                optimizer.zero_grad()
                batch_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                    
            active_tokens = (decoder_out != PAD_IDX).sum().item()
            accumulated_loss += batch_loss.item() * active_tokens
            accumulated_tokens += active_tokens
            
    mean_loss = accumulated_loss / max(accumulated_tokens, 1)
    return mean_loss

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int = 3,
    device: str = "cpu",
) -> torch.Tensor:
    """Unrolls sequence token by token at test time."""
    model.eval()
    with torch.no_grad():
        encoded_memory = model.encode(src, src_mask)
        generated_seq = torch.tensor([[start_symbol]], dtype=torch.long, device=device)
        
        for _ in range(max_len - 1):
            current_mask = make_tgt_mask(generated_seq, pad_idx=PAD_IDX).to(device)
            step_predictions = model.decode(encoded_memory, src_mask, generated_seq, current_mask)
            next_token = step_predictions[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_seq = torch.cat([generated_seq, next_token], dim=1)
            
            if next_token.item() == end_symbol:
                break
                
    return generated_seq

def _corpus_bleu(hypotheses, references, max_order=4) -> float:
    match_counts = [0] * max_order
    total_ngrams = [0] * max_order
    len_ref = 0
    len_hyp = 0
    
    for hyp, refs in zip(hypotheses, references):
        len_hyp += len(hyp)
        best_ref_len = min((len(r) for r in refs), key=lambda x: (abs(x - len(hyp)), x), default=0)
        len_ref += best_ref_len
        
        for n in range(1, max_order + 1):
            hyp_grams = [tuple(hyp[i:i+n]) for i in range(len(hyp) - n + 1)]
            total_ngrams[n-1] += len(hyp_grams)
            
            hyp_freqs = Counter(hyp_grams)
            ref_max_freqs = Counter()
            
            for ref in refs:
                ref_grams = [tuple(ref[i:i+n]) for i in range(len(ref) - n + 1)]
                for g, count in Counter(ref_grams).items():
                    ref_max_freqs[g] = max(ref_max_freqs.get(g, 0), count)
                    
            for g, count in hyp_freqs.items():
                match_counts[n-1] += min(count, ref_max_freqs.get(g, 0))
                
    precisions = []
    for i in range(max_order):
        if total_ngrams[i] > 0:
            p = match_counts[i] / total_ngrams[i]
            if p == 0: p = 0.1 / total_ngrams[i]
        else:
            p = 1e-3
        precisions.append(p)
        
    geom_mean = math.exp(sum((1.0/max_order) * math.log(p) for p in precisions))
    bp = math.exp(1.0 - len_ref/len_hyp) if len_hyp > 0 and len_hyp < len_ref else (0.0 if len_hyp == 0 else 1.0)
    return bp * geom_mean * 100.0

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """Computes BLEU metrics on a defined DataLoader instance."""
    model.eval()
    
    def fetch_token_str(idx):
        try:
            if hasattr(tgt_vocab, "lookup_token"): return tgt_vocab.lookup_token(idx)
            if hasattr(tgt_vocab, "get_itos"): return tgt_vocab.get_itos()[idx]
        except: pass
        return str(idx)
        
    hypothesis_list = []
    reference_list = []
    ignore_indices = {PAD_IDX, SOS_IDX, EOS_IDX}
    
    with torch.no_grad():
        for source_b, target_b in test_dataloader:
            source_b = source_b.to(device)
            target_b = target_b.to(device)
            
            for i in range(source_b.size(0)):
                src_item = source_b[i:i+1]
                tgt_item = target_b[i]
                
                mask_enc = make_src_mask(src_item, pad_idx=PAD_IDX).to(device)
                decoded_output = greedy_decode(model, src_item, mask_enc, max_len, SOS_IDX, EOS_IDX, device)
                
                hyp_indices = decoded_output.squeeze(0).tolist()
                if EOS_IDX in hyp_indices: hyp_indices = hyp_indices[:hyp_indices.index(EOS_IDX)]
                hyp_indices = [idx for idx in hyp_indices if idx not in ignore_indices]
                
                ref_indices = tgt_item.tolist()
                if EOS_IDX in ref_indices: ref_indices = ref_indices[:ref_indices.index(EOS_IDX)]
                ref_indices = [idx for idx in ref_indices if idx not in ignore_indices]
                
                hypothesis_list.append([fetch_token_str(idx) for idx in hyp_indices])
                reference_list.append([[fetch_token_str(idx) for idx in ref_indices]])
                
    return _corpus_bleu(hypothesis_list, reference_list, max_order=4)

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """Serialises standard torch training artefacts to disk."""
    checkpoint_data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
    }
    torch.save(checkpoint_data, path)

def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    loaded_data = torch.load(path, map_location="cpu")
    is_dict = isinstance(loaded_data, dict)
    state_d = loaded_data.get("model_state_dict", loaded_data) if is_dict else loaded_data.state_dict()
    
    model.load_state_dict(state_d, strict=False)
    
    if optimizer and is_dict and "optimizer_state_dict" in loaded_data:
        try: optimizer.load_state_dict(loaded_data["optimizer_state_dict"])
        except: pass
        
    if scheduler and is_dict and "scheduler_state_dict" in loaded_data:
        try: scheduler.load_state_dict(loaded_data["scheduler_state_dict"])
        except: pass
        
    return loaded_data.get("epoch", 0) if is_dict else 0

def run_training_experiment() -> None:
    pass

if __name__ == "__main__":
    run_training_experiment()