"""
test.py

Executable script.
1. Downloads best pretrained checkpoint from Google Drive.
2. Initializes dataloaders and model schema.
3. Injects weights and runs for `N` additional epochs.
"""

import torch
import torch.optim as optim
import gdown
import os
from dataset import build_dataloaders
from model import Transformer
from train import run_epoch, LabelSmoothingLoss, save_checkpoint
from lr_scheduler import NoamScheduler

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing over device engine: {device}")
    
    print("Binding data loader engines...")
    train_dl, val_dl, test_dl, src_v, tgt_v = build_dataloaders(batch_sz=64)
    
    src_vocab_size = len(src_v)
    tgt_vocab_size = len(tgt_v)
    
    print("Constructing Transformer...")
    # These base dimensions align with the parameters used to train the base model instance on GDrive.
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=256,
        N=3,
        num_heads=8,
        d_ff=512,
        dropout=0.1
    ).to(device)
    
    checkpoint_file = "best_pretrained_ckpt.pt"
    
    # Attempting cloud download of model baseline parameters
    if not os.path.exists(checkpoint_file):
        print("Obtaining pre-trained weights from remote index...")
        gdown.download(id="12ii8FI5fcp91bwVvYEUwjbExj2hiN_bc", output=checkpoint_file, quiet=False)
        
    if os.path.exists(checkpoint_file):
        print("Injecting static weights into Transformer model...")
        ckpt = torch.load(checkpoint_file, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt.state_dict()
        model.load_state_dict(state_dict, strict=False)
        print("Model state resolved. Starting epochs...")
    
    optimizer = optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, d_model=256, warmup_steps=4000)
    loss_criterion = LabelSmoothingLoss(vocab_size=tgt_vocab_size, pad_idx=1, smoothing=0.1).to(device)
    
    num_epochs = 3
    print(f"Beginning continuous training trajectory for {num_epochs} Epochs...")
    for epoch in range(num_epochs):
        train_loss = run_epoch(train_dl, model, loss_criterion, optimizer, scheduler, epoch, True, device)
        val_loss = run_epoch(val_dl, model, loss_criterion, None, None, epoch, False, device)
        
        print(f"Epoch Index {epoch+1}/{num_epochs} \n -> Train Loss: {train_loss:.4f} \n -> Val Loss: {val_loss:.4f}")
        
        save_path = f"checkpoint_epoch_{epoch+1}.pt"
        save_checkpoint(model, optimizer, scheduler, epoch+1, save_path)
        print(f"Persisted parameters to: {save_path}")

if __name__ == "__main__":
    main()