from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import torch
 
class MaskDataset(Dataset):
    # root dir is the folder directory with the images
    def __init__ (self, root_dir, csv_file, transform = None):
        self.root_dir = root_dir
        self.csv = pd.read_csv(csv_file)
        self.transform = transform
 
    # return the length of the csv file which is also the length of the Dataset
    def __len__ (self):
        return len(self.csv)
 
    # want to return an (x,y) pair --> (image, label)
    def __getitem__(self, index):
        # [index, 0] because index gives the row that the image's info is saved and 0 is the column with the image names
        image_name = self.csv.iloc[index, 0]
        # need .convert bcs PIL Image open doesn't give an RGB, so you need to convert it
        image = Image.open(os.path.join(self.root_dir, image_name)).convert("RGB")
 
        # get the label
        # want it to be a tensor (torch.tensor()) bcs all data must be in tensor before going into CNN
        # use float and tensor to meet loss function requirements (won't use float here specifically bcs it gives errors in the model)
        y = torch.tensor(self.csv.iloc[index,1]) # (float(self.csv.iloc[index,1]))
 
        # if there is a transform defined, apply the transform
        if self.transform is not None:
            image = self.transform(image)
    
        # returns the image and its label
        return (image, y)