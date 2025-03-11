import torch
from src.cropsAndWeedsSegmentation.utils.model_eval_utils import pixel_wise_acc
from src.cropsAndWeedsSegmentation.constants import DEVICE

def train_fn(data_loader,model,optimizer):
    model.train()
    total_loss, pixel_acc = 0.0,0.0
    for images,masks in data_loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE).squeeze(1).long()

        optimizer.zero_grad()
        logits,loss = model(images,masks)
        loss.backward()
        optimizer.step()

        total_loss+=loss.item()
        pixel_acc += pixel_wise_acc(logits,masks)
    return total_loss/len(data_loader), pixel_acc/len(data_loader)
        
## Eval function
def eval_fn(data_loader,model):
    model.eval()
    total_loss, pixel_accuracy = 0.0,0.0
    with torch.no_grad():
        for images,masks in data_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE).squeeze(1).long()
            logits,loss = model(images,masks)
            total_loss+=loss.item()
            pixel_accuracy += pixel_wise_acc(logits,masks)
            
        return total_loss/len(data_loader), pixel_accuracy/len(data_loader)
        
