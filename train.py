import os
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from FACodec_AC.config import Config
from huggingface_hub import hf_hub_download
import sys
from FACodec_AC.dataset import ZContentDataset, LengthSortedBatchSampler, collate_fn_zcontent
from FACodec_AC.utils import init_facodec_models
from FACodec_AC.models import DenoisingTransformerModel 

SCRIPT_LOCATION = os.environ.get("location")
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Amphion'))


def lr_lambda(step):
    warmup_steps = 1000
    if step < warmup_steps:
        return step / warmup_steps
    return 1.0  # constant after warmup

def main():
    # Seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    
    # Initialize FACodec models.
    fa_encoder, fa_decoder = init_facodec_models(Config.device)
    # It is needed only in get_item in the Dataset for looking up indexes, and workers work on CPU anyway.
    fa_decoder = fa_decoder.to('cpu') 
    # Extract codebooks and projection layers from the content quantizer.
    codebook_zc1 = fa_decoder.quantizer[1].layers[0].codebook.weight
    out_proj_zc1 = fa_decoder.quantizer[1].layers[0].out_proj
    codebook_zc2 = fa_decoder.quantizer[1].layers[1].codebook.weight
    out_proj_zc2 = fa_decoder.quantizer[1].layers[1].out_proj

    # Use ZContentDataset instead of CodebookSequenceDataset.
    train_dataset = ZContentDataset(
        os.path.join(Config.facodec_dataset_dir, 'train'),
        os.path.join(Config.phoneme_cond_dir,'train'),
        codebook_zc1, out_proj_zc1,
        codebook_zc2, out_proj_zc2,
        facodec_dim=Config.FACodec_dim
    )
    test_dataset  = ZContentDataset(
        os.path.join(Config.facodec_dataset_dir, 'test'),
        os.path.join(Config.phoneme_cond_dir,'test'),
        codebook_zc1, out_proj_zc1,
        codebook_zc2, out_proj_zc2,
        facodec_dim=Config.FACodec_dim
    )
    
    # Use LengthSortedBatchSampler for bucketing.
    train_batch_sampler = LengthSortedBatchSampler(train_dataset, batch_size=Config.batch_size, 
                                                   drop_last=False, 
                                                   shuffle=True,
                                                   batches_per_bucket=Config.batches_per_bucket)
    test_batch_sampler  = LengthSortedBatchSampler(test_dataset, batch_size=Config.batch_size, 
                                                   drop_last=False,
                                                   shuffle=False,
                                                   batches_per_bucket=Config.batches_per_bucket)
    
    dataloader_train = DataLoader(train_dataset, batch_sampler=train_batch_sampler, collate_fn=collate_fn_zcontent)
    dataloader_test  = DataLoader(test_dataset, batch_sampler=test_batch_sampler, collate_fn=collate_fn_zcontent)
    
    # Initialize the transformer model.
    model = DenoisingTransformerModel(
        d_model=Config.d_model,
        nhead=Config.nhead,
        num_layers=Config.num_layers,
        d_ff=Config.d_ff,
        dropout=Config.dropout,
        max_seq_len=Config.max_seq_len,
        FACodec_dim=Config.FACodec_dim,
        phone_vocab_size=Config.PHONE_VOCAB_SIZE,
        num_steps=Config.num_steps,  # number of diffusion steps
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr, betas=(0.9, 0.95), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    model.to(Config.device)

    # -- Initialize from Checkpoint if available --
    if os.path.exists(Config.checkpoint_path):
        print(f"Loading checkpoint from {Config.checkpoint_path}")
        checkpoint = torch.load(Config.checkpoint_path)
        model.load_state_dict(checkpoint)
    model.train()
    
    writer = SummaryWriter(log_dir=Config.tensorboard_dir)
    best_eval_loss = float('inf')
    global_step = 0

    for epoch in range(Config.epochs):
        total_loss = 0.0
        total_loss_zc1 = 0.0
        total_loss_zc2 = 0.0
        num_batches = 0

        # --- Training ---
        for zc1, zc2, phone_cond, mask in dataloader_train:
            global_step += 1
            optimizer.zero_grad()
            x0 = zc1.to(Config.device)
            zc2 = zc2.to(Config.device)
            padded_phone_ids = phone_cond.to(Config.device)
            padding_mask = mask.to(Config.device)
            bsz, feature_dim, seq_len = x0.shape
            
            # --- diffusion corruption ---
            # sample a timestep t for each example
            t = torch.randint(0, model.num_steps, (bsz,), device=Config.device)
            # compute √ᾱ_t and √(1−ᾱ_t)
            sa  = model.sqrt_abar[t].view(bsz, 1, 1)
            s1a = model.sqrt_1mabar[t].view(bsz, 1, 1)
            # draw noise ε
            eps = torch.randn_like(x0)
            # form noisy input z_t
            x_noisy = sa * x0 + s1a * eps
            
            # forward pass: predict ε and zc2
            eps_pred, zc2_pred = model(
                zc1_noisy=x_noisy,
                padded_phone_ids=padded_phone_ids,
                t=t,
                padding_mask=padding_mask
            )
            
            # ε-loss (MSE)
            mask_exp = (~padding_mask).unsqueeze(1)  # [B,1,T]

            loss_zc1 = F.mse_loss(eps_pred, eps, reduction='none')
            loss_zc1 = (loss_zc1).masked_select(mask_exp).mean()

            loss_zc2 = F.mse_loss(zc2_pred, zc2, reduction='none')
            loss_zc2 = loss_zc2.masked_select(mask_exp).mean()
            loss = loss_zc1 + 0.5*loss_zc2
            loss.backward()
            grad_norm = 0#torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=30.0)
            # optional: peek at the raw norm every few steps
            if global_step % 100 == 0:
                print(f"step {global_step:>7d} |grad| after clip = {grad_norm:6.3f}")
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            total_loss_zc1 += loss_zc1.item()
            total_loss_zc2 += loss_zc2.item()
            num_batches += 1

            if global_step % 20 == 0:
                current_lr = scheduler.get_last_lr()[0]
                writer.add_scalar("Diag/grad_norm", grad_norm, global_step)
                writer.add_scalar("Diag/lr", current_lr, global_step)

        avg_loss = total_loss / max(num_batches, 1)
        avg_loss_zc1 = total_loss_zc1 / max(num_batches, 1)
        avg_loss_zc2 = total_loss_zc2 / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{Config.epochs}, Loss={avg_loss:.4f}, zc1 Loss={avg_loss_zc1:.4f}, zc2 Loss={avg_loss_zc2:.4f}")
        writer.add_scalar("Loss/Train_zc1", avg_loss_zc1, epoch+1)
        writer.add_scalar("Loss/Train_zc2", avg_loss_zc2, epoch+1)

        # --- Evaluation ----------------------------------------------------------
        if (epoch + 1) % Config.eval_epochs == 0:
            model.eval()
            total_test_loss = 0.0
            total_test_loss_zc1 = 0.0
            total_test_loss_zc2 = 0.0
            test_batches = 0

            with torch.no_grad():
                for zc1_val, zc2_val, test_phone_ids, mask in dataloader_test:
                    x0 = zc1_val.to(Config.device)
                    zc2_val = zc2_val.to(Config.device)
                    padded_phone_ids = test_phone_ids.to(Config.device)
                    padding_mask = mask.to(Config.device)
                    bsz, feature_dim, seq_len = x0.shape

                    t = torch.randint(0, model.num_steps, (bsz,), device=Config.device)

                    # compute √ᾱ_t and √(1−ᾱ_t)
                    sa  = model.sqrt_abar[t].view(bsz, 1, 1)
                    s1a = model.sqrt_1mabar[t].view(bsz, 1, 1)
                    # draw noise ε
                    eps = torch.randn_like(x0)
                    # form noisy input z_t
                    x_noisy = sa * x0 + s1a * eps

                    # forward pass: predict ε and zc2
                    eps_pred, zc2_pred = model(
                        zc1_noisy=x_noisy,
                        padded_phone_ids=padded_phone_ids,
                        t=t,
                        padding_mask=padding_mask
                    )

                    mask_exp = (~padding_mask).unsqueeze(1)  # [B,1,T]
                    loss_zc1 = F.mse_loss(eps_pred, eps, reduction='none')
                    loss_zc1 = (loss_zc1).masked_select(mask_exp).mean()

                    loss_zc2 = F.mse_loss(zc2_pred, zc2_val, reduction='none')
                    loss_zc2 = loss_zc2.masked_select(mask_exp).mean()
                    loss = loss_zc1 + 0.5*loss_zc2


                    total_test_loss_zc1  += loss_zc1.item()
                    total_test_loss_zc2  += loss_zc2.item()
                    total_test_loss += loss.item()
                    test_batches += 1

            avg_test_loss = total_test_loss / max(test_batches, 1)
            avg_test_loss_zc1 = total_test_loss_zc1 / max(test_batches, 1)
            avg_test_loss_zc2 = total_test_loss_zc2 / max(test_batches, 1)

            print(f"Epoch {epoch+1}/{Config.epochs}, Eval Loss={avg_test_loss:.4f}, zc1 Loss={avg_test_loss_zc1:.4f}, zc2 Loss={avg_test_loss_zc2:.4f}")
            writer.add_scalar("Loss/Eval_zc1", avg_test_loss_zc1, epoch+1)
            writer.add_scalar("Loss/Eval_zc2", avg_test_loss_zc2, epoch+1)

            # Save checkpoint only if current eval loss is lower than best so far and epoch has passed Config.checkpoint_epochs.
            if avg_test_loss < best_eval_loss:
                best_eval_loss = avg_test_loss
                checkpoint_full_path = Config.checkpoint_path
                checkpoint_dir = os.path.dirname(checkpoint_full_path)
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_full_path)
                msg = f"Epoch {epoch+1}: New best eval loss: {avg_test_loss:.4f}. Checkpoint saved at {checkpoint_full_path}"
                print(msg)
                writer.add_text("Training/Checkpoint", msg, global_step=epoch+1)
            model.train()
    
    writer.close()

if __name__ == "__main__":
    main()