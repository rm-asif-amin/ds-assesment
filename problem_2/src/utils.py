import matplotlib.pyplot as plt
import numpy as np
import random
import os
import torch
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def seed_worker(worker_id,random_seed=0):
    worker_seed = random_seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  

def save_model(model, out_folder: str):
    """
    Saves serialised model to an output folder. 
    Parameters:
    model : trained pytorch model
    out_folder: path to output folder.
    Returns:
    Nothing
    """
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
        
    model_name_on_disk='trained_model.pt'
    model_save_path=os.path.join(out_folder, model_name_on_disk)
    torch.save(model,model_save_path)

def get_test_report(test_labels,test_preds,classes):
    test_labels=[x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x for x in test_labels]
    test_preds=[x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x for x in test_preds]

    test_labels=np.concatenate(test_labels).ravel()
    test_preds=np.concatenate(test_preds).ravel()

    return classification_report(test_labels, test_preds, target_names=classes)

def plot_metrics(history):
    dir_name=os.path.join(os.path.dirname(os.path.realpath(__file__)),'plots')

    for metric in ['accuracy','loss']:
        plt.figure(figsize=(10,5))
        plt.title("Training and Validation "+ metric)


        val_hist=[x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in history['val'][metric]]
        train_hist=[x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in history['train'][metric]]


        plt.plot(val_hist,label="validation")
        plt.plot(train_hist,label="train")

        plt.xlabel("iterations")
        plt.ylabel(metric)
        plt.legend()

        file_name="train_val_"+metric+".png"

        if not os.path.exists(dir_name):
          os.makedirs(dir_name)
        plt.savefig(os.path.join(dir_name,file_name))

