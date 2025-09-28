import model_dataset as Dataset
import numpy as np
import time
import render
import threading
import sys

size_of_dataset = 103

label_dataset = np.empty(size_of_dataset-1, dtype="int")
point_cloud_dataset = np.empty(size_of_dataset-1, dtype="object")

for i in range(size_of_dataset-1):
    model = Dataset.load_data(f"Voxel_Dataset_100p/model{i}.npy")
    point_cloud = Dataset.convert_to_point_cloud(model)  

    point_cloud_dataset[i] = point_cloud
    print(f"{i+1} of {size_of_dataset}")

class ShowingModel(threading.Thread):
    def __init__(self, point_cloud):
        threading.Thread.__init__(self)
        self._stop_event = threading.Event()
        self.point_cloud = point_cloud

    def run(self):
        render.show_model(point_cloud)

class TakeInput(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.results = None
    
    def run(self):
        label = input("Label: ")
        self.results = label

for i, point_cloud in enumerate(point_cloud_dataset):

    print(point_cloud)

    thread1 = ShowingModel(point_cloud)
    thread2 = TakeInput()

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()

    label_dataset[i] = thread2.results

np.save("Label.npy", label_dataset)

