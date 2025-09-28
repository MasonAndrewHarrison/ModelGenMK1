import model_dataset as Dataset
import numpy as np
import time
import render


flower_dataset = Dataset.PointCloud()
point_cloud_list = flower_dataset.load_data('normalized_dataset.npy')

start = time.perf_counter()

for index, m in enumerate(point_cloud_list, start=82):

    print(f"{index} out of 103")
    model = point_cloud_list[index]

    model = Dataset.converter_to_voxel(model, 100, 100, 100)

    directory = f"Voxel_Dataset_100p/model{index}.npy"
    Dataset.save_data(directory, model)

end = time.perf_counter()

print(end - start)


