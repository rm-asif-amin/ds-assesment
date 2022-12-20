import copy
from tqdm import tqdm
import time
import torch
import matplotlib.pyplot as plt
from utils import imshow,get_test_report,plot_metrics

class Model:
    def __init__(self,pretrained_model,device):
        self.model=pretrained_model
        self.device = device

    def train(self, train_val_dataloaders,optimizer,scheduler,loss_fn,dataset_sizes,num_epochs=25):
        since = time.time()
        history={'train':{'loss':[],
                      'accuracy':[]},
                'val':{'loss':[],
                      'accuracy':[]}}

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 20)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in tqdm(train_val_dataloaders[phase]):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = loss_fn(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                
                history[phase]['loss'].append(epoch_loss)
                history[phase]['accuracy'].append(epoch_acc)
                
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            print()

        time_elapsed = time.time() - since
        print(f'Training time required {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best validation Accuracy: {best_acc:4f}')

        plot_metrics(history)
        
        # load best model weights
        self.model.load_state_dict(best_model_wts)
        return self.model , history
    
    def evaluate(self,test_dataloader,loss_fn,classes,test_datasize):

        print(f'Evaluating Test Data')
        print('-' * 20)

        self.model.eval()

        test_labels,test_preds=[],[]

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(test_dataloader):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                   
                    # forward
                    with torch.no_grad():
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = loss_fn(outputs, labels)

                        
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    test_labels.append(labels)
                    test_preds.append(preds)

        test_loss = running_loss / test_datasize
        test_acc = running_corrects.double() / test_datasize

        print(f'Test Loss: {test_loss:.4f},Test Acc: {test_acc:.4f}')

        test_report=get_test_report(test_labels,test_preds,classes)
        print(test_report)
       


                

    