import fire
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from model import Model
from torchvision import  models
from preprocessing import Preprocessor
import os
from utils import save_model

def train(dataset_path: str='/opt/applications/image_classifier/dataset', saved_model_folder: str='/opt/applications/image_classifier/saved_model',evaluate_model: bool=True ,epochs : int=25, lr: float=0.001,learning_rate_scheduler : str='step', eps: float=1e-08) -> None:

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  
  preprocessor=Preprocessor(dataset_path=dataset_path)
  train_dataloader,val_dataloader, dataset_sizes,classes=preprocessor.train_load_transform(train_val_split=0.8)
  num_classes=len(classes)
  train_val_dataloaders={'train':train_dataloader,'val':val_dataloader}
  
  model_ft = models.resnet18(pretrained=True)
  num_ftrs = model_ft.fc.in_features
  model_ft.fc = nn.Linear(num_ftrs, num_classes)

  model_ft = model_ft.to(device)

  loss_metric = nn.CrossEntropyLoss() 
  optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr, momentum=0.9)

  # Decay LR by a factor of 0.1 every 7 epochs  
  exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


  derived_model = Model(pretrained_model=model_ft,device=device)
  trained_model,history=derived_model.train(train_val_dataloaders, optimizer=optimizer_ft,scheduler=exp_lr_scheduler ,loss_fn=loss_metric,dataset_sizes= dataset_sizes,
                       num_epochs=epochs)

  save_model(trained_model,out_folder=saved_model_folder)
  
  if evaluate_model:
     test_dataloader,test_datasize=preprocessor.test_load_transform()
     derived_model.evaluate(test_dataloader,loss_metric,classes,test_datasize)

  save_model(trained_model,out_folder=saved_model_folder)

if __name__ == '__main__':
  fire.Fire(train)