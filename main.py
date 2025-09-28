import model_dataset as Dataset
import numpy as np
import time
import render
import threading

label_dataset = np.array([])

class ShowingModel(threading.Thread):
    def __init__(self, point_cloud):
        threading.Thread.__init__(self)
        self.point_cloud = point_cloud

    def run(self):
        render.show_model(point_cloud)

class TakeInput(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.results = None
    
    def run(self):
        self.results = 2

for i in range(103):

    model = Dataset.load_data(f"Voxel_Dataset_100p/model{i}.npy")
    point_cloud = Dataset.convert_to_point_cloud(model)

    thread1 = ShowingModel(point_cloud)
    thread2 = TakeInput()

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()

    print(thread2.results)


