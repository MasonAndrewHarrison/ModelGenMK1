import model_dataset as Dataset
import numpy as np
import time
import render
import threading
import sys

label_dataset = np.load('Label.npy')

for i in range(102):

    model = Dataset.load_data(f"Voxel_Dataset_32p/model{i}.npy")
    point_cloud = Dataset.convert_to_point_cloud(model)  

    print(label_dataset[i])
    render.show_model(point_cloud)