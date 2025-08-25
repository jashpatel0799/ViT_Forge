# main.py
"""
Main entry for curriculum learning with a fixed 100-class ViT head.

Curriculum (in order, each for 10 epochs by default unless overridden in config):
  1) MNIST
  2) EMNIST (balanced, 47 classes)
  3) FashionMNIST
  4) SVHN
  5) CIFAR10
  6) CIFAR100
Final evaluation:
  - STL10 test set

Key features:
  - Single model through all stages (no head swapping).
  - Slice logits to active dataset class count during loss/accuracy calculation.
  - W&B logging per stage.
  - Save a checkpoint *after each stage*.
"""
import argparse
import yaml
import os
import torch
import torch.nn as nn
import random
import numpy as np
import data, model, train, utils
import wandb


# SEED = 64
# NUM_EPOCH = 1
# LEARNING_RATE = 1e-3
# INCHANNEL = 3
# IMG_SIZE = 224
# PATCH_SIZE = 16
# EMBED_SIZE = (PATCH_SIZE ** 2) * 3
# NUM_HEADS = 8
# DEPTH = 12
# BATCH_SIZE = 8
# NUM_CLASSES = 10
# POS_EMB_TYPE = "linear"

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False

def main(cfgs: dict):
    
   print("\n")
   print(f"Experiment Name: {cfgs['exp_name']}")
   print(f"Experiment Model Number: {cfgs['model_num']}")
   print(f"Optimizer used: {cfgs['optimizer_name']}")
   print(f"Experiment Details: {cfgs['details']}")
   print("\n")
   print(f"Dataset Name: {cfgs['dataset_name']}")
   print(f"Seed: {cfgs['seed']}")
   print(f"Number of Epochs: {cfgs['num_epoch']}")
   print(f"Learning Rate: {cfgs['learning_rate']}")
   print(f"Input Channel: {cfgs['input_channel']}")
   print(f"Input Image Size: {cfgs['input_image_size']}")
   print(f"Patch Size: {cfgs['patch_size']}")
   print(f"Embedding Size: {(cfgs['patch_size'] ** 2) * 3}")
   print(f"Number of Heads: {cfgs['num_head']}")
   print(f"ViT Deptlinearh: {cfgs['vit_depth']}")
   print(f"Batch Size: {cfgs['batch_size']}")
   print(f"Number of Classes: {cfgs['num_class']}")
   print(f"curriculum: {cfgs['curriculum']}")
   print(f"Position Embed Use: {cfgs['pos_emb_type']}")
   print(f"WandB Project: {cfgs['wandb_project']}")
   print(f"WandB Run Name: {cfgs['wandb_runname']}")
   print(f"Output Directory: {cfgs['output_dir']}")
   print("\n")
   
   
   # SEED = 64
   # NUM_EPOCH = 1
   # LEARNING_RATE = 1e-3
   # INCHANNEL = 3
   # IMG_SIZE = 224
   # PATCH_SIZE = 16
   # EMBED_SIZE = (PATCH_SIZE ** 2) * 3
   # NUM_HEADS = 8
   # DEPTH = 12
   # BATCH_SIZE = 8
   # NUM_CLASSES = 10
   # POS_EMB_TYPE = "linear"
   
   # HYPERPARAMETERS
   # ------------------------------ Config ----------------------------------------
   # Expect these keys in your config.yml (with defaults for safety):
   seed = int(cfg.get("seed", 42))
   exp_name        = cfgs.get("exp_name", "vit_curriculum")
   dataset_root    = cfgs.get("dataset_root", "./data")
   image_size      = int(cfgs.get("input_image_size", 96))
   batch_size      = int(cfgs.get("batch_size", 64))
   learning_rate   = float(cfgs.get("learning_rate", 1e-3))
   weight_decay    = float(cfgs.get("weight_decay", 0.03))
   num_heads       = int(cfgs.get("num_head", 8))
   patch_size      = int(cfgs.get("patch_size", 16))
   in_channels     = int(cfgs.get("input_channel", 3))
   embed_size      = (patch_size ** 2) * in_channels
   vit_depth       = int(cfgs.get("vit_depth", 12))
   pos_type        = cfgs.get("pos_emb_type", "linear")
   output_dir      = cfgs.get("output_dir", "./checkpoints")
   os.makedirs(output_dir, exist_ok=True)
   
   # Curriculum stages (name, epochs). You asked: 10 each.
   # You can override in config with `curriculum` list if you prefer.
   curriculum = cfgs.get("curriculum", [
      {"name": "mnist",        "epochs": 10},
      {"name": "emnist",       "epochs": 25},
      {"name": "fashionmnist", "epochs": 15},
      {"name": "svhn",         "epochs": 25},
      {"name": "cifar10",      "epochs": 25},
      {"name": "cifar100",     "epochs": 25},
   ])
   
   device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

   set_seed(seed)

   # ------------------------------ Model -----------------------------------------
   # IMPORTANT: Fix classifier head to 100 (max classes). Keep this consistent!
   fixed_num_classes = 100
   vit_model = model.vit(in_channel=in_channels, img_size=image_size, patch_size=patch_size, 
                     num_heads=num_heads, embedding_size=embed_size, batch_size=batch_size, 
                     depth=vit_depth, num_class=fixed_num_classes, pos_type=pos_type).to(device)

   loss_fn = torch.nn.CrossEntropyLoss()
   # optimizer = torch.optim.SGD(vit_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
   optimizer = torch.optim.AdamW(vit_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
   
   # ------------------------------ W&B -------------------------------------------
   # Single run across the entire curriculum, stage metrics are prefixed (e.g., "MNIST/Train Loss")
   wandb_project = cfgs.get("wandb_project", "vit_curriculum")
  #  wandb_runname = cfgs.get("wandb_runname", f"{exp_name}_lr{learning_rate}_d{vit_depth}_p{patch_size}")
   wandb_runname = f"{exp_name}_pos-{pos_type}_lr{learning_rate}_d{vit_depth}_p{patch_size}"
   run = wandb.init(project=wandb_project, name=wandb_runname, config=cfgs)
   
   for stage in curriculum:
      stage_name = stage["name"]
      stage_epochs = int(stage.get("epochs", 10))

      print(f"\n=== Stage: {stage_name.upper()} | Epochs: {stage_epochs} ===")
      train_dataloader, test_dataloader, num_current_classes = data.prepare_dataloader(
         dataset_name=stage_name,
         batch_size=batch_size,
         image_size=image_size,
         root=dataset_root,
      )
      print(f"{stage_name}: num_classes = {num_current_classes}")

      # Train+eval on this dataset
      metrics = train.train(
         stage_name=stage_name.upper(),
         model=vit_model,
         train_dataloader=train_dataloader,
         test_dataloader=test_dataloader,
         optimizer=optimizer,
         loss_fn=loss_fn,
         device=device,
         epochs=stage_epochs,
         num_current_classes=num_current_classes,
         wandb_run=run,
      )
      
      # Save a checkpoint after each stage
      ckpt_name = f"vit_{stage_name}_ep{stage_epochs}.pth"
      utils.save_model(model=vit_model, traget_dir=output_dir, model_name=ckpt_name)
      
   # ------------------------------ Final Evaluation on STL10 ---------------------
   print("\n=== Final Evaluation on STL10 (test) ===")
   _, stl_test_loader, stl_num_classes = data.prepare_dataloader(
      dataset_name="stl10",
      batch_size=batch_size,
      image_size=image_size,
      root=dataset_root,
   )
   # one pass eval
   test_loss, test_acc = train.eval_loop(
      model=vit_model,
      dataloader=stl_test_loader,
      loss_fn=loss_fn,
      device=device,
      num_current_classes=stl_num_classes,  # slice to 10
   )
   print(f"[STL10] Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

   if run is not None:
      run.log({
         "STL10/Test Loss": test_loss,
         "STL10/Test Acc":  test_acc
      })
      run.finish()
      
   
    
if __name__ == "__main__":
   parser = argparse.ArgumentParser(description="ViT Curriculum Learning")
   parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
   args = parser.parse_args()

   with open(args.config, "r") as f:
      cfg = yaml.safe_load(f)

   # IMPORTANT: Your model already prints EMBED SIZE using (patch_size**2)*3.
   # Keep input_channel=3 in your config so ViT receives 3-channel images.

   main(cfg)
