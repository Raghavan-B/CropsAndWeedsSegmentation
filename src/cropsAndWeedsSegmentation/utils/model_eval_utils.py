import time
import torch

def pixel_wise_acc(output, target):
    '''
    '''
    # Get predicted class indices
    pred = torch.argmax(output, dim=1)  # Shape: (batch, H, W)

    # Remove channel dimension from target if needed
    if target.dim() == 4:  # Shape: (batch, 1, H, W) -> (batch, H, W)
        target = target.squeeze(1)

    # Ensure target is integer type
    target = target.long()

    # Compute pixel-wise accuracy
    correct = (pred == target).float().sum()
    total = target.numel()

    return (correct / total).item()

def get_model_size(model):
    '''
    '''
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())  # Size of parameters
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())  # Size of buffers
    total_size = (param_size + buffer_size) / (1024 ** 2)  # Convert to MB
    return total_size

def inference_speed(image,model,num_trails = 50):
    '''
    '''
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        for _ in range(num_trails):
            _ = model(image)
    end_time = time.time()
    avg_time = (end_time-start_time)/num_trails
    return avg_time
