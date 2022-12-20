from torchvision import datasets,transforms
import os
from torch.utils.data import DataLoader,random_split
import torch



class Preprocessor:
    def __init__(self,dataset_path,device="cpu"):
        self.dataset_path=dataset_path
        self.device=device

    
    def get_transforms(self):
        train_transforms = transforms.Compose([   
        transforms.ToTensor()
            ])

        train_data_transforms =  transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(policy = transforms.autoaugment.AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        test_data_transforms=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        return train_data_transforms,test_data_transforms

    def train_load_transform(self,train_val_split=0.8):
        train_data_transforms,_=self.get_transforms()
        
        train_data=datasets.ImageFolder(os.path.join(self.dataset_path, 'train'),train_data_transforms)
        classes = train_data.classes

        g = torch.Generator()
        g.manual_seed(0)

        train_data,val_data=random_split(train_data, [train_val_split, 1-train_val_split], generator=g)



        train_dataloader=DataLoader(train_data, batch_size=4, shuffle=True, num_workers=8)
        val_dataloader=DataLoader(val_data, batch_size=4, shuffle=True, num_workers=8)

        dataset_sizes={'train':len(train_data),
                        'val': len(val_data)}

        
        
        return train_dataloader,val_dataloader, dataset_sizes,classes
        
    def test_load_transform(self):
        _,test_data_transforms=self.get_transforms()
        
        test_data= datasets.ImageFolder(os.path.join(self.dataset_path, 'test'),test_data_transforms)
        test_dataloader=DataLoader(test_data, batch_size=16, shuffle=True, num_workers=8)

        dataset_size=len(test_data)

        return test_dataloader,dataset_size