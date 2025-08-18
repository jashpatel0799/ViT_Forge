import torch
from timeit import default_timer as timer
from tqdm.auto import tqdm
import wandb


def train_loop(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module, accuracy_fn, 
               device: torch.device):
    """
    training loop for pytorch model

    Args:
        model (torch.nn.Module): Pytorch model to train
        dataloader (torch.utils.data.DataLoader): train set dataloader
        optimizer (torch.optim.Optimizer): optimizer to optimize the model
        loss_fn (torch.nn.Module): loss function to calcualte loss
        accuracy_fn (_type_): accuracy function tomeasure the accuracy
        device (torch.device): device on which weight are load
        
    Return: 
        train_model: train model weights
        train_loss: loss value
        train_acc: accuract\y value
        
    Eample:
        train_loop(model=model, dataloader=train_dataloader, 
                    optimizer=optimizer, loss_fn=loss_fn, 
                    accuracy_fn=accuracy_fn, device=device)
    """
    
    train_loss, train_acc = 0, 0
    
    model.train()
    
    for batch, (x_train, y_train) in enumerate(dataloader):
        # print("\nBATCH:", batch)
        x_train, y_train = x_train.to(device), y_train.to(device)
        
        # 1. Forward step
        train_pred = model(x_train)
        
        # 2. Loss 
        loss = loss_fn(train_pred, y_train)
        
        # 3. Grad stepzero
        optimizer.zero_grad()
        
        # 4. Backward
        loss.backward()
        
        # 5. Optimizer step
        optimizer.step()
        
        acc = accuracy_fn(torch.argmax(train_pred,dim=1), y_train)
        
        train_acc += acc
        train_loss += loss
        # print("ACC:", acc)
        # print("Loss:", loss)
        
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    
    return train_loss, train_acc, model

def test_loop(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module, accuracy_fn, device: torch.device):
    
    """
    testing loop for pytorch model

    Args:
        model (torch.nn.Module): Pytorch model to test
        dataloader (torch.utils.data.DataLoader): test set dataloader
        loss_fn (torch.nn.Module): loss function to calcualte loss
        accuracy_fn (_type_): accuracy function tomeasure the accuracy
        device (torch.device): device on which weight are load
        
    Return: 
        test_loss: loss value
        test_acc: accuract\y value
        
    Eample:
        test_loop(model=model, dataloader=test_dataloader, 
                  loss_fn=loss_fn, accuracy_fn=accuracy_fn, device=device)
    """
    
    test_loss, test_acc = 0, 0
    
    model.eval()
    with torch.inference_mode(): #with torch.no_grad:
        for x_test, y_test in dataloader:
            
            x_test, y_test = x_test.to(device), y_test.to(device)
            
            test_pred = model(x_test)
            
            loss = loss_fn(test_pred, y_test)
            
            acc = accuracy_fn(torch.argmax(test_pred, dim=1), y_test)
            
            test_loss += loss
            test_acc += acc
            
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
        
    return test_loss, test_acc
        
        
        

def train(model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module, accuracy_fn, epoches: int, device: torch.device, args):
    
    """
    Train and test the model
    
    Args: 
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
        
    Example:
        train(model = model_0, train_dataloader = traindataloader, test_dataloader = testdataloader,
              optimizer = optim, loss_fn = lossfunction, accuracy_fn = accuracyfunction, epoches=n, device=device)
    """
    
    wandb.init(project=args['wandb_project'], name=args['wandb_runname'], config=args)
    
    train_losses, train_acces = [], []
    test_losses, test_acces = [], []
    
    for epoch in tqdm(range(epoches)):
        
        train_loss, train_acc, train_model = train_loop(model=model, dataloader=train_dataloader, 
                                                   optimizer=optimizer, loss_fn=loss_fn, 
                                                   accuracy_fn=accuracy_fn, device=device)
        
        
        test_loss, test_acc = test_loop(model=model, dataloader=test_dataloader, loss_fn=loss_fn,
                                   accuracy_fn=accuracy_fn, device=device)
        
        
        wandb.log({
            "Train Loss": train_loss,
            "Test Loss": test_loss,
            "Train Accuracy": train_acc,
            "Test Accuracy": test_acc
        })
        
        print(f"\nEpoch: {epoch+1}/[{epoches+1}] ")
        print(f"Train Loss: {train_loss:.5f}  Test Loss: {test_loss:.5f}  ||  Train Accuracy:  {train_acc:.5f}  Test Accuracy: {test_acc:.5f}")
        
        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())
        train_acces.append(train_acc.item())
        test_acces.append(test_acc.item())
        
    wandb.finish()
        
    return train_model, train_losses, test_losses, train_acces, test_acces