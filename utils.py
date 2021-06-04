import numpy as np
import torch
import torch.nn.functional as F

def error(mem):
    if len(mem) < 100:
        return 1.0
    all_vals = (np.cumsum(mem, 0)/np.reshape(np.arange(1, len(mem)+1), (-1,1)))[-100:]
    errors = np.mean(np.abs(all_vals[-100:] - all_vals[-1:])/(np.abs(all_vals[-1:]) + 1e-12), -1)
    errors = np.max(errors)
    print('Error: {}'.format(errors))
    return errors

def accuracy(logits, labels):
    correct, total = 0, 0
    with torch.no_grad():
        preds = F.softmax(logits, dim=1)
        predicted = torch.argmax(preds, dim=1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    return correct / total