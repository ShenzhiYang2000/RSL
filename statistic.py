import numpy as np
import torch
import pandas as pd

def record(features, g,  model, epoch):
    with torch.no_grad():

        model.eval()

        pred, _ = model(features, g, use_h = True)
        # softmax = torch.nn.Softmax(dim=1)
        # pred = softmax(pred)
        pred = np.array((pred[:, 0]).cpu())
        # if epoch == 0:
        #     preds = pred
        # else:
        #     preds = np.concatenate([preds, pred], axis=0)

           

    return  pred





def three_sigma(x):

    # idx = np.where(x < 0.2/14)
    idx = np.where(x < 0.2 / 9)
    return x[idx]
