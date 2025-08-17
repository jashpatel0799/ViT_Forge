import argparse
import yaml
import torch
import torch.nn as nn
from torchmetrices.classification import MulticlassAccuracy
import random
import numpy as np
import data, model, train, utils
import wandb
from torchsummary import summary


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

def main(args):
    
    print("\n")
    print(f"Experiment Name: {args['exp_name']}")
    print(f"Experiment Model Number: {args['model_num']}")
    print(f"Optimizer used: {args['optimizer_name']}")
    print(f"Experiment Details: {args['details']}")
    print("\n")
    print(f"Dataset Name: {args['dataset_name']}")
    print(f"Seed: {args['seed']}")
    print(f"Number of Epochs: {args['num_epoch']}")
    print(f"Learning Rate: {args['learning_rate']}")
    print(f"Input Channel: {args['input_channel']}")
    print(f"Input Image Size: {args['input_image_size']}")
    print(f"Patch Size: {args['patch_size']}")
    print(f"Embedding Size: {(args['patch_size'] ** 2) * 3}")
    print(f"Number of Heads: {args['num_head']}")
    print(f"ViT Depth: {args['vit_depth']}")
    print(f"Batch Size: {args['batch_size']}")
    print(f"Number of Classes: {args['num_class']}")
    print(f"Position Embed Use: {args['pos_emb_type']}")
    print(f"WandB Project: {args['wandb_project']}")
    print(f"WandB Run Name: {args['wandb_runname']}")
    print(f"Output Directory: {args['output_dir']}")
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
    EXP_NAME = args['exp_name']
    MODEL_NUM = args['model_num']
    OPTIMIZER = args['optimizer_name']
    DATASET = args['dataset_name']
    EXP = EXP_NAME + "_" + DATASET
    SEED = args['seed']
    NUM_EPOCH = args['num_epoch']
    LEARNING_RATE = float(args['learning_rate'])#3e-4 # 3e-4, 4e-5, 7e-6, 5e-7, 3e-9
    INCHANNEL = args['input_channel']
    IMG_SIZE = args['input_image_size']
    PATCH_SIZE = args['patch_size']
    EMBED_SIZE = (PATCH_SIZE ** 2) * INCHANNEL # args['embedding_size']  
    NUM_HEADS = args['num_head']
    DEPTH = args['vit_depth']
    POS_EMB_TYPE = args['pos_emb_type']
    BATCH_SIZE = args['batch_size']
    NUM_CLASSES = args['num_class']
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


    vit_model = model.vit(in_channel=INCHANNEL, img_size=IMG_SIZE, patch_size=PATCH_SIZE, 
                        num_heads=NUM_HEADS, embedding_size=EMBED_SIZE, batch_size=BATCH_SIZE, 
                        depth=DEPTH, num_class=NUM_CLASSES, pos_type=POS_EMB_TYPE).to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss()
    accuracy_fn = MulticlassAccuracy(num_classes = NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.SGD(vit_model.parameters(), lr=LEARNING_RATE, weight_decay=0.03)

    train_model, train_loss, test_loss, train_acc, test_acc = train.train(model=vit_model,
                                                                        train_dataloader=data.train_dataloader,
                                                                        test_dataloader=data.test_dataloader,
                                                                        optimizer=optimizer,
                                                                        loss_fn=loss_fn,
                                                                        accuracy_fn=accuracy_fn,
                                                                        epoches=NUM_EPOCH,
                                                                        device=DEVICE, args=args)

    utils.save_model(model=train_model, traget_dir="./save_model", 
                    model_name=f"vit_model_{POS_EMB_TYPE}.pth")

    utils.plot(train_losses=train_loss, test_losses=test_loss, train_accs=train_acc, 
            test_accs=test_acc, fig_name=f"loss_and_accuracy_{POS_EMB_TYPE}.jpg")
    
    
if __name__ == "__main__":
   parser = argparse.ArgumentParser(description="Original Architecture of VIT")
   parser.add_argument("--config", type=str, required=True, help="Path to the config file")
   
   args = parser.parse_args()
   
   # Load config file
   with open(args.config, 'r') as file:
      config = yaml.safe_load(file)

   # Automatically generate wandb_runname
   config['wandb_runname'] = f"{config['exp_name']}_{config['dataset_name']}_Lr_{config['learning_rate']}_EMB_{(config['patch_size'] ** 2) * 3}_patch_{config['patch_size']}_depth_{config['vit_depth']}_pos_emb_type_{config['pos_emb_type']}"
   
   main(config)
