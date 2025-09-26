import model_dataset as Dataset
import numpy as np
import time
import render



model = Dataset.load_data("Voxel_Dataset/model2.npy")


point_cloud = Dataset.convert_to_point_cloud(model)
render.show_model(point_cloud)

