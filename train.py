"""
Training/evaluation loops that support:
- Fixed classifier head (100 logits) with *per-stage* class slicing.
- W&B logging per stage (e.g., "MNIST/Train Loss").
- Checkpoint saving hook (handled in main).
"""

import torch
from torch import nn
from tqdm.auto import tqdm
import wandb
from typing import Dict, Tuple, Optional

@torch.no_grad()
def _compute_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    """
    Simple top-1 accuracy (avoid extra deps)
    """
    preds =logits.argmax(dim=1)
    correct = (preds == target).sum().item()
    return correct / target.numel()

def train_loop(model: nn.Module, 
               dataloader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer, 
               loss_fn: nn.Module, 
               device: torch.device,
               num_current_classes: int
) -> Tuple[float, float]:
    """
    Train model for one epoch on the current dataset subset of classes.
    Only the first `num_classes_current` logits are supervised.

    Args:
        model (torch.nn.Module): Pytorch model to train
        dataloader (torch.utils.data.DataLoader): train set dataloader
        optimizer (torch.optim.Optimizer): optimizer to optimize the model
        loss_fn (nn.Module): loss function to calcualte loss
        device (torch.device): device on which weight are load
        num_current_classes (int): number of classes in curent dataset
        
    """
    
    train_loss, train_acc, n_batches = 0, 0, 0
    
    model.train()
    
    for batch, (x_train, y_train) in enumerate(dataloader):
        # print("\nBATCH:", batch)
        x_train, y_train = x_train.to(device, non_blocking = True), y_train.to(device, non_blocking = True)
        
        # 1. Forward step
        train_pred = model(x_train)
        train_logits = train_pred[:, :num_current_classes]
        
        # 2. Loss 
        loss = loss_fn(train_logits, y_train)
        
        # 3. Grad stepzero
        optimizer.zero_grad(set_to_none=True)
        
        # 4. Backward
        loss.backward()
        
        # 5. Optimizer step
        optimizer.step()
        
        acc = _compute_accuracy(train_logits, y_train)
        
        train_acc += acc
        train_loss += loss.item()
        n_batches += 1
        # print("ACC:", acc)
        # print("Loss:", loss)
        
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    
    return train_loss / max(1, n_batches), train_acc / max(1, n_batches)

@torch.no_grad()
def eval_loop(model: nn.Module, 
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module, 
              device: torch.device,
              num_current_classes: int
) -> Tuple[float, float]:
    
    """
    Evaluate model for one pass on the current dataset test split,
    slicing logits to the active class count.

    Args:
        model (torch.nn.Module): Pytorch model to test
        dataloader (torch.utils.data.DataLoader): test set dataloader
        loss_fn (torch.nn.Module): loss function to calcualte loss
        device (torch.device): device on which weight are load
        num_current_classes (int): number of classes in curent dataset
        
    """
    
    eval_loss, eval_acc, n_batches = 0, 0, 0
    
    model.eval()
    with torch.inference_mode(): #with torch.no_grad:
        for x_eval, y_eval in dataloader:
            
            x_eval, y_eval = x_eval.to(device, non_blocking=True), y_eval.to(device, non_blocking=True)
            
            eval_pred = model(x_eval)
            
            eval_logits = eval_pred[:, :num_current_classes]
            
            loss = loss_fn(eval_logits, y_eval)
            
            acc = _compute_accuracy(eval_logits, y_eval)
            
            eval_loss += loss
            eval_acc += acc
            n_batches += 1
            
        eval_loss /= len(dataloader)
        eval_acc /= len(dataloader)
        
    return eval_loss / max(1, n_batches), eval_acc / max(1, n_batches)
        
        
        

def train(stage_name: str,
          model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module, 
          epochs: int, 
          device: torch.device,
          num_current_classes: int, 
          wandb_run = None
) -> Dict[str, float]:
    
    """
    Train for `epochs` on a single dataset (stage). Logs per-epoch metrics.
    Returns final metrics for the stage.
    
    Args: 
        stage_name: in curriculam learning to track the DB
        model: A pytorch model to be train and test
        train_dataloader: a dataloader instance to train a model
        test_dataloader: a dataloader instance to test a model
        optimizer: A optimizer to optimize a model
        loss_fn : A loss function to calculate loss on both dataloader
        accuracy_fn : A accuracy function to measure the accuracy
        epoch: inter get to train how may number of epochs to train the model
        device: a device on which a model to be train and test
        
    Return: 
        list of train and test model loss and accuracy 
        train also return model weights
        
    
    """
    
    best_test_acc = 0.0
    last_metrics: Dict[str, float] = {}
    
    for epoch in tqdm(range(1, epochs+1)):
        
        train_loss, train_acc = train_loop(model=model, dataloader=train_dataloader, 
                                            optimizer=optimizer, loss_fn=loss_fn, device=device, 
                                            num_current_classes=num_current_classes)
        
        
        eval_loss, eval_acc = eval_loop(model=model, dataloader=test_dataloader, loss_fn=loss_fn,
                                        device=device, num_current_classes=num_current_classes)
        
        if wandb_run is not None:
            wandb_run.log({
                f"{stage_name}/Train Loss": train_loss,
                f"{stage_name}/Test Loss": eval_loss,
                f"{stage_name}/Train Accuracy": train_acc,
                f"{stage_name}/Test Accuracy": eval_acc,
                f"{stage_name}/Epoch": epoch
            })
            
        if eval_acc > best_test_acc:
            best_test_acc = eval_acc
        
        print(f"\n[{stage_name}] Epoch: {epoch}/[{epochs}] \tTrain Loss: {train_loss:.5f} Test Loss: {eval_loss:.5f}  ||  Train Accuracy:  {train_acc:.5f}  Test Accuracy: {eval_acc:.5f}")
        
        last_metrics = {
            "train_acc": train_acc,
            "train_loss": train_loss,
            "test_acc": eval_acc,
            "test_loss": eval_loss,
            "best_test_acc": best_test_acc
        }
        
        
    return last_metrics