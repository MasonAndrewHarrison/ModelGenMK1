import model_dataset as Dataset
import numpy as np
import time
import render


for i in range(103):
    model = Dataset.load_data(f"Voxel_Dataset_100p/model{i}.npy")


    point_cloud = Dataset.convert_to_point_cloud(model)
    render.show_model(point_cloud)

