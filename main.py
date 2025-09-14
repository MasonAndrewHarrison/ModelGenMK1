import model_dataset as Dataset
import numpy as np
import time


flower_dataset = Dataset.PointCloud()
point_cloud_list = flower_dataset.load_data('normalized_dataset.npy')
my_model = point_cloud_list[32]

print(3)
voxel_dataset = Dataset.VoxelMatrix(
    point_cloud=my_model,
    x_size=4,
    y_size=4,
    z_size=4
)

voxel_dataset.save_data()

'''
voxel_dataset = Dataset.VoxelMatrix(
    directory="Voxel_Dataset/Model1.npy"
)


model = voxel_dataset.get_model()
voxel_dataset.convert_to_point_cloud()
'''