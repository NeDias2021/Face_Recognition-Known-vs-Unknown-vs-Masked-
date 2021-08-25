import pandas as pd
import os
import torch
 
# device configuration
# have the GPU support
device = ("cuda" if torch.cuda.is_available() else "cpu")
 
# create a DataFrame with columns image_name and label
train_df = pd.DataFrame(columns=["image_name","label"])
 
# os.listdir returns a list of the files in the directory in the parameter
# the rows under image_name will now have all of the image file names
train_df["image_name"] = os.listdir("/root/project/train")
 
# now for each file (the point of idx is it is going through each image so all of the rows)
# name is the image name
# if the image has no_mask in the name then the label column of that image's row will get a 0 for False
# if the image has mask in the name then the label column of that image's row will get a 1 for True
for idx, name in enumerate(os.listdir("/root/project/train")):
    if "mask" in name or "Mask" in name:
        train_df["label"][idx] = 1
    else:
        train_df["label"][idx] = 0
 
# convert the DataFrame into a csv
    # the csv will be called train_csv
    # index = False means you don't want to write row names
    # header = True means you want to write the column names (image_name and label)
train_df.to_csv (r'train_csv.csv', index = False, header=True)