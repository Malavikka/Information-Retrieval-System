# %%
# import required modules
import pandas as pd
import numpy as np
import os
# %%
# retrive all files names
files_list = []
for i,j,k in os.walk("./Processed_dataset/"):
    files_list.extend(k)
print("Number of Datasets : ",len(files_list))
# %%
# merge all csv files and add origin file column just for reference if needed
# add docID column for the merged dataset, to give unique docID for each document
path_prefix = "./Processed_dataset/"
merged_dataset = pd.read_csv(path_prefix+files_list[0])
x = [files_list[0]+'-'+str(i) for i in merged_dataset.index]
merged_dataset["Origin_file"] = x
for fname in files_list[1:]:
    df = pd.read_csv(path_prefix+fname)
    x = [fname+'-'+str(i) for i in df.index]
    df["Origin_file"] = x
    merged_dataset = pd.concat([merged_dataset,df],ignore_index=True)
merged_dataset.drop(columns=["Unnamed: 0"],inplace=True)
merged_dataset["docID"] = merged_dataset.index
merged_dataset.to_csv("Merged_Dataset.csv",index=False)
merged_dataset
# %%
