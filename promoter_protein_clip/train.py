import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import wandb
import argparse
import os
from promoter_protein_clip.dataset import get_collate_fn
from promoter_protein_clip.model_transformer import SequenceCLIPTransformer
from promoter_protein_clip.dataset import get_hf_sequence_dataset


def get_args():
    parser = argparse.ArgumentParser(description="Train a SequenceCLIP model.")

    # data
    parser.add_argument('--data_path', type=str, default='data/toy_data.csv',
                        help="Path to the input CSV file (e.g., toy_data.csv).")
    parser.add_argument('--max_promoter_length', type=int, default=512,
                        help="Maximum length of promoter sequences.")  
    parser.add_argument('--protein_max_len', type=int, default=1024,
                        help="Maximum length of protein sequences.")      
    parser.add_argument('--val_data_path', type=str, default=None)
    parser.add_argument('--train_embed_path', type=str, default=None)
    parser.add_argument('--train_mapping_path', type=str, default=None)
    parser.add_argument('--val_embed_path', type=str, default=None)
    parser.add_argument('--val_mapping_path', type=str, default=None)
    
    # Augmentation
    parser.add_argument('--random_shift', action='store_true', default=False)
    parser.add_argument('--random_shift_min', type=int, default=99)
    
    # Intervals
    parser.add_argument('--validation_interval', type=int, default=-1,
                        help="Other than at epoch end, how often to get validation loss. -1 for only on epoch end.")
    parser.add_argument('--log_interval', type=int, default=100,
                        help="How often to log")    
    
    # model
    parser.add_argument('--protein_model', type=str, default='facebook/esm2_t6_8M_UR50D',
                        help="Name of the pre-trained ESM model to use.")
    parser.add_argument('--projection_dim', type=int, default=256,
                        help="Shared embedding dimension for projection heads.")
    parser.add_argument('--transformer_depth', type=int, default=6,
                        help="Number of layers in transformer model") 
    parser.add_argument('--transformer_num_heads', type=int, default=8,
                        help="Number of attention heads per layer in transformer model")   
    parser.add_argument('--promoter_embedding_dim', type=int, default=256,
                        help="Size of hidden dimension in promoter encoder") 
    parser.add_argument('--unfreeze', action='store_true', default=False,
                        help="Unfreeze the protein encoder, I hope you have some time to spare")                

    # training
    parser.add_argument('--epochs', type=int, default=10,
                        help="Total number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=512,
                        help="Batch size for training and validation.")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="Learning rate for the optimizer.")
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help="Weight decay for the optimizer.")
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help="Number of warmup steps for the learning rate scheduler.")  
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help="Maximum gradient norm for gradient clipping.")
    # checkpoint
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help="Directory to save model checkpoints.")
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help="Path to a checkpoint file to resume training from.")
    # W&B logging
    parser.add_argument('--wandb_project', type=str, default='SequenceCLIP',
                        help="Name of the Weights & Biases project.")
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help="Your W&B entity (username or team).")
    parser.add_argument('--wandb_run_name', type=str, default='sequence-clip',
                        help="Name of the Weights & Biases run.")
                      

    return parser.parse_args()
                  

def train_epoch(model, loader, optimizer, device, epoch_num, val_loader=None, args=None, scheduler=None):
    global_step = (epoch_num - 1) * len(loader)
    """Performs one training epoch."""
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch_num}/{args.epochs} [Train]")
    for protein_strings, promoter_strings in pbar:         
        global_step += 1
                
        if isinstance(protein_strings, dict): # If not using pre-embedded protein we can
            protein_strings = {k: v.to(device, non_blocking=True) for k, v in protein_strings.items()}
        else:
            protein_strings = protein_strings.to(device, non_blocking=True)
        promoter_strings = promoter_strings.to(device, non_blocking=True)        
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            model_outs = model(protein_strings, promoter_strings)
            loss = model_outs['loss']

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        batch_loss = loss.item()
        total_loss += batch_loss

        pbar.set_postfix(loss=batch_loss)
        
        # Unneccessary to log batch loss every stop
        if global_step % args.log_interval == 0:
            wandb.log({
            "train/batch_loss": batch_loss,
            "train/learning_rate": scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr'],
            "global_step": global_step,
            })

        # If you want to run validation more than at epoch end  
        if args.validation_interval > 0 and global_step % args.validation_interval == 0:            
            
            val_loss = validate_epoch(
                model, val_loader, epoch_num, args.epochs, device,global_step=global_step
            )

            wandb.log({
                "val/step_loss": val_loss,
                "global_step": global_step,
            })       
                                          
    avg_loss = total_loss / len(loader)        
    val_loss = validate_epoch(
        model, val_loader, epoch_num, args.epochs, device,global_step=global_step
    )
    wandb.log({
        "val/epoch_loss": val_loss,
        "train/epoch_loss": avg_loss,
        "epoch": epoch_num,
        "global_step": global_step,
        "learning_rate": optimizer.param_groups[0]['lr'],
    })
    
    return avg_loss,val_loss,global_step

def validate_epoch(model, loader, epoch_num, total_epochs, device,global_step=None):
    """Performs one validation epoch."""
    model.eval()
    total_loss = 0.0
    
    desc_suffix = f" (Step {global_step})" if global_step is not None else ""
    pbar = tqdm(loader, desc=f"Epoch {epoch_num}/{total_epochs} [Validate]{desc_suffix}")    

    with torch.no_grad():
        for protein_strings, promoter_strings in pbar:  
            
            if isinstance(protein_strings, dict): #If not using precomputed embeddings
                protein_strings = {k: v.to(device, non_blocking=True) for k, v in protein_strings.items()}
            else:
                protein_strings = protein_strings.to(device, non_blocking=True)
            promoter_strings = promoter_strings.to(device, non_blocking=True)                

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                model_outs = model(protein_strings, promoter_strings)
                loss = model_outs['loss']
            
            batch_loss = loss.item()
            total_loss += batch_loss

            pbar.set_postfix(val_loss=batch_loss)

    avg_loss = total_loss / len(loader)
    return avg_loss

def main():
    args = get_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Initializing W&B...")
    wandb.init(
        name=args.wandb_run_name,
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args),
    )

    print("Initializing SequenceCLIP model...")         
    model = SequenceCLIPTransformer(
        protein_model_name=args.protein_model,
        projection_dim=args.projection_dim,
        device=device,
        max_promoter_length=args.max_promoter_length,
        protein_max_len=args.protein_max_len,
        train_embeds_dict=args.train_embed_path,
        val_embeds_dict=args.val_embed_path,
        promoter_embedding_dim=args.promoter_embedding_dim,
        depth=args.transformer_depth,
        num_heads=args.transformer_num_heads,
        unfreeze=args.unfreeze,
    )                      

    model = model.to(device)
    model = torch.compile(model)

    print(f"Loading data from {args.data_path}...")

    train_dataset = get_hf_sequence_dataset(csv_file_path=args.data_path, max_promoter_length=args.max_promoter_length, random_shift=args.random_shift, random_shift_min=args.random_shift_min)
    val_dataset = get_hf_sequence_dataset(csv_file_path=args.val_data_path, max_promoter_length=args.max_promoter_length, random_shift_min=args.random_shift_min)
    print(f"Total samples: {len(train_dataset)+ len(val_dataset)}, Train: {len(train_dataset)}, Validation: {len(val_dataset)}")        

    if args.train_mapping_path is not None:
        collate_train = get_collate_fn(model.protein_tokenizer,  model.promoter_tokenizer, model.protein_max_len, args.train_embed_path, args.train_mapping_path)  #TODO: Make max protein length an argument
    else:
        collate_train = get_collate_fn(model.protein_tokenizer,  model.promoter_tokenizer, model.protein_max_len, args.train_embed_path)  #TODO: Make max protein length an argument
    if args.val_mapping_path is not None:
        collate_val = get_collate_fn(model.protein_tokenizer,  model.promoter_tokenizer, model.protein_max_len, args.val_embed_path, args.val_mapping_path)        
    else:
        collate_val = get_collate_fn(model.protein_tokenizer,  model.promoter_tokenizer, model.protein_max_len, args.val_embed_path)        

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_train)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_val)
    
    wandb.watch(model, log="all", log_freq=8000) # log gradients, parameters, is this needed?

    params = list(model.parameters())
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    num_training_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args.warmup_steps, 
        num_training_steps=num_training_steps,
    )

    start_epoch = 1
    best_val_loss = float('inf')

    if args.resume_checkpoint:
        if os.path.exists(args.resume_checkpoint):
            print(f"Resuming training from checkpoint: {args.resume_checkpoint}")
            checkpoint = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))

            print(f" -> Resuming from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
        else:
            print(f"Warning: Checkpoint path not found, starting from scratch: {args.resume_checkpoint}")

    print("--- Starting Training ---")
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, val_loss, final_global_step = train_epoch(
            model, train_loader, optimizer, device, epoch, val_loader=val_loader, args=args, scheduler=scheduler
        )

        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")        
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            best_model_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'args': args,
            }, best_model_path)            
            
            print(f" -> New best model saved to {best_model_path}")

            wandb.save(best_model_path)        

        # checkpoint saving
        latest_checkpoint_path = os.path.join(args.save_dir, "latest_checkpoint.pth")
        latest_model_path = os.path.join(args.save_dir, f"model_epoch_{epoch}.pth")
        #torch.save(model.state_dict(), latest_model_path)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'args': args,
        }, latest_model_path)        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'args': args,
        }, latest_checkpoint_path)

        wandb.save(latest_checkpoint_path)

    print("--- Training Complete ---")
    wandb.finish()

if __name__ == "__main__":
    main()