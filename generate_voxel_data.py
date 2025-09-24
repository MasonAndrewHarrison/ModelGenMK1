import model_dataset as Dataset
import numpy as np
import time
import render


flower_dataset = Dataset.PointCloud()
point_cloud_list = flower_dataset.load_data('normalized_dataset.npy')

for index, model in enumerate(point_cloud_list):

    print(f"{index} out of 103")

    model = Dataset.converter_to_voxel(model, 2, 2, 2)

    directory = f"Voxel_Dataset_2p/model{index}.npy"
    Dataset.save_data(directory, model)
