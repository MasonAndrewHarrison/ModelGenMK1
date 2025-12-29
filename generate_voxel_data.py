import model_dataset as Dataset
import numpy as np
import time
import render
import os

resolution = 32
starting = 6

os.makedirs(f'Voxel_Dataset_{resolution}p', exist_ok=True)

flower_dataset = Dataset.PointCloud()
point_cloud_list = flower_dataset.load_data('normalized_dataset.npy')

start = time.perf_counter()

for index,_ in enumerate(point_cloud_list, start=starting):

    print(f"{index+1} out of 103")
    model = point_cloud_list[index]

    model = Dataset.converter_to_voxel(model, resolution, resolution, resolution)

    directory = f"Voxel_Dataset_{resolution}p/model{index}.npy"
    Dataset.save_data(directory, model)

end = time.perf_counter()

print(end - start)


