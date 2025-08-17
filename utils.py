import torch
import matplotlib.pyplot as plt
from pathlib import Path

def save_model(model: torch.nn.Module, traget_dir: str, model_name: str):
    """
    Save the train model weights

    Args:
        model (torch.nn.Module): model weights which you want to save
        traget_dir (str): laction wher you want to save model weights
        model_name (str): model name vith extention
        
    Example: 
        save_model(model=model_0, traget_dir = dir_path, model_name = "modl.pth")
    """
    
    traget_dir_path = Path(traget_dir)
    traget_dir_path.mkdir(parents=True, exist_ok=True)
    
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model name should be ends with pt or pth"
    model_save_path = traget_dir_path / model_name
    
    print(f"\n Saving Model At: {model_save_path}")
    torch.save(obj = model.state_dict(), f= model_save_path)
    

def load_model(model: torch.nn.Module, model_path : str):
    """
    Load the already train model weights
        
    Args:
        model (torch.nn.Module): model wich need to load
        model_path (str): path where model weight is save
        
    Example:
        load_model(model=model_0, model_path = "models/model.pth")
    """
    model.load_state_dict(torch.load(f = model_path, map_location=torch.device('cpu')))
    print(f"\nModel Loaded.")
    
    
def plot(train_losses: list[float], test_losses: list[float], train_accs: list[float], test_accs: list[float], fig_name: str):
    """
    plot loss and accuarcy of train and test data
    Args:
        train_losses (list[float]): list of train losses
        test_losses (list[float]): list of test losses
        train_accs (list[float]): list of train accuracy
        test_accs (list[float]): list of tst accuracy
        fig_name (str): name by which you want to save plt image 
        
    Example:
        plot(train_losses=train_loss, test_losses=test_loss, train_accs=train_acc, test_accs=test_acc, fig_name="loss/accuracy.png")
    """
    plt.figure(figsize=(20,8))
    plt.subplot(1,2,1)
    plt.plot(range(len(train_losses)), train_losses, label = "Train Loss")
    plt.plot(range(len(test_losses)), test_losses, label = "Test Loss")
    plt.legend()
    plt.xlabel("Epoches")
    plt.ylabel("Loss")
    
    plt.subplot(1,2,2)
    plt.plot(range(len(train_accs)), train_accs, label = "Train Accuracy")
    plt.plot(range(len(test_accs)), test_accs, label = "Test Accuracy")
    plt.legend()
    plt.xlabel("Epoches")
    plt.ylabel("Accuracy")
    
    plt.savefig(fig_name)