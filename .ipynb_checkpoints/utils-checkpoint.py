"""
Creates various utility functions for PyTorch mdoel training and testing
"""

import torch
import torch.nn as nn
from pathlib import Path

def accuracy(output, target):
    # Get the index of max log probability
    pred = output.max(1, keep_dim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()

def save_model(model: nn.Module,
               target_dir: str, 
               model_name: str):
    """
        Saves a PyTorch model to target directory.
        
        Args:
        model: A PyTorch model to save.
        target_dir: A destination directory.
        model_name: A filename for the saved model. Should include either ".pth" or ".pt" as the file extension.
        
        Example usage:
            save_model(model=model_0,
                       target_dir="model5"
                       model_name="going_modular.pth"
                       
    """
    
    # Create target directory:
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)
    
    # Create model save path 
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'."
    model_save_path = target_dir_path / model_name
    
    # Save the model state_dict()
    print(f"[INFO] saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)
    
    