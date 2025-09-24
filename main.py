import model_dataset as Dataset
import numpy as np
import time
import render


flower_dataset = Dataset.PointCloud()
point_cloud_list = flower_dataset.load_data('normalized_dataset.npy')
my_model = point_cloud_list[92]


model = Dataset.converter_to_voxel(my_model, 100, 100, 100)

Dataset.save_data("Voxel_Dataset/model2.npy", model)
model = Dataset.load_data("Voxel_Dataset/model2.npy")


point_cloud = Dataset.convert_to_point_cloud(model)
render.show_model(point_cloud)

