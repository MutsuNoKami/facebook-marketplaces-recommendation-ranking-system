
import torch    
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.merged_data = pd.read_csv('training_data.csv') 
        self.files = self.merged_data['image_id']
        self.labels = self.merged_data['labels']
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        
        self.num_classes = len(set(self.labels))
        self.decoder = pd.read_pickle('image_decoder.pkl')
        self.transform = transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(128),
                transforms.CenterCrop(128),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]) 
            ])


    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        label = self.labels[index]
        label = torch.as_tensor(label).long()
        image = Image.open('cleaned_images/' + self.files[index] + '.jpg')
        if image.mode != 'RGB':
            image = self.transform(image).float()
        else:
            image = self.transform(image)
        return image, label


