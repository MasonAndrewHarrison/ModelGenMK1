import model_dataset as Dataset
import numpy as np
import time
import render


flower_dataset = Dataset.PointCloud()
point_cloud_list = flower_dataset.load_data('normalized_dataset.npy')
my_model = point_cloud_list[92]


voxel_dataset = Dataset.VoxelMatrix(
    directory="Voxel_Dataset/Model2.npy",
    point_cloud=my_model,
    x_size=200,
    y_size=200,
    z_size=200
)

voxel_dataset.save_data()


voxel_dataset = Dataset.VoxelMatrix(
    directory="Voxel_Dataset/Model2.npy"
)


model = voxel_dataset.get_model()
point_cloud = voxel_dataset.convert_to_point_cloud()
print(point_cloud)
render.show_model(point_cloud)

