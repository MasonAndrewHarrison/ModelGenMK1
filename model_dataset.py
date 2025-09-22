from plyfile import PlyData
import numpy as np
from typing import overload
import bisect
import pandas as pd
import time
from functools import reduce
import render as Visualizer
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import Pool
import os
import copy
from joblib import Parallel, delayed
from tqdm import tqdm

class PointCloud:

    def __init__(self, **kwargs):

        self.default_path = kwargs.get('default_path', "")
        self.size_of_dataset = kwargs.get('size_of_dataset', 0)
        self.is_normalized = False

        if self.default_path == "":
            self.model_list = np.empty(0, dtype='object')

        elif kwargs.get('normalize', True):
            self.model_list = self._get_all_models()
            self.normalize_all()

        else:
            self.model_list = self._get_all_models()


    def save_data(self, save_directory):
        
        np.save(save_directory, self.model_list)


    def load_data(self, load_directory) -> np.ndarray:

        self.model_list = np.load(load_directory, allow_pickle=True)
        self.size_of_dataset = self.model_list.shape[0]
    
        return self.model_list


    def __add__(self, other) -> None:
        
        if isinstance(other, Dataset):
            return self.get_all_models() + other.getAllModels()

        raise TypeError("Unsupported type")   


    def _normalize(self, index) -> None:

        point_cloud = self.model_list[index]

        x_mean = point_cloud[:, 0].mean()
        y_mean = point_cloud[:, 1].mean()
        z_mean = point_cloud[:, 2].mean()

        point_cloud[:, 0] -= x_mean
        point_cloud[:, 1] -= y_mean
        point_cloud[:, 2] -= z_mean 

        max_value = point_cloud[:, :3].max()
        point_cloud[:, :3] /= max_value * 2

        point_cloud[:, :3] += 0.5

        
    def normalize_all(self) -> None:

        normalize = self._normalize
        for i in range(self.size_of_dataset): normalize(i)
        self.is_normalized = True


    def _get_model(self, index) -> np.ndarray:

        index += 1
        if index > self.size_of_dataset or index < 0: 
            
            raise IndexError(f"File index {index} is out of bounds. Valid range is 0-{self.size_of_dataset-1}")

        file = f'{self.default_path}Model{index}'
        plydata = PlyData.read(file)
        vertex_list = plydata['vertex'].data

        field_names = vertex_list.dtype.names 
        vertex_array = np.array([vertex_list[field] for field in field_names], dtype='float16')

        normalized_vertex_list = vertex_array.T
        normalized_vertex_list[:, 3:6] /= 255

        return normalized_vertex_list


    def get_model(self, index):
        
        if index >= self.size_of_dataset or index < 0:   
            raise IndexError(f"File index {index} is out of bounds. Valid range is 0-{self.size_of_dataset-1}")
        
        else: return self.model_list[index]

    
    def _get_all_models(self):

        model_list = np.empty(self.size_of_dataset, dtype='object')
        for i,_ in enumerate(model_list):
            model_list[i] = self._get_model(i)

        return model_list


    def get_all_models(self):

        return self.model_list

    
    @overload
    def show_model(self, index: int) -> np.ndarray: pass

    @overload
    def show_model(self, model: list) -> None: pass

    def show_model(self, modelKey: int | np.ndarray):

        if isinstance(modelKey, int):

            index = modelKey
            model = self.get_model(index)
            Visualizer.show_model(model)
            return model

        elif isinstance(modelKey, np.ndarray):

            model = modelKey
            Visualizer.show_model(model)  

        else: raise TypeError("Unsupported type")      


class VoxelMatrix:

    def __init__(self, **kwargs):

        if 'point_cloud' in kwargs:
            self.point_cloud = kwargs['point_cloud']
            
            self.x_size = kwargs['x_size']
            self.y_size = kwargs['y_size']
            self.z_size = kwargs['z_size']

            self.sorted_x_indices = np.argsort(self.point_cloud[:, 0])
            self.sorted_y_indices = np.argsort(self.point_cloud[:, 1])
            self.sorted_z_indices = np.argsort(self.point_cloud[:, 2])

            self.sorted_x_pc = self.point_cloud[self.sorted_x_indices]
            self.sorted_y_pc = self.point_cloud[self.sorted_y_indices]
            self.sorted_z_pc = self.point_cloud[self.sorted_z_indices]

            self.extracted_sorted_x_pc = [point[0] for point in self.sorted_x_pc]
            self.extracted_sorted_y_pc = [point[1] for point in self.sorted_y_pc]
            self.extracted_sorted_z_pc = [point[2] for point in self.sorted_z_pc]

    def get_points_in_range(self, **kwargs):

        dimension_key = kwargs['dimension_key']
        lowest_range = kwargs['start']
        highest_range = kwargs['end']

        sorted_point_cloud = None
        extracted_sorted_pc = None

        if dimension_key == 'x':
            sorted_point_cloud = self.sorted_x_pc
            extracted_sorted_pc = self.extracted_sorted_x_pc

        elif dimension_key == 'y':
            sorted_point_cloud = self.sorted_y_pc
            extracted_sorted_pc = self.extracted_sorted_y_pc

        elif dimension_key == 'z':
            sorted_point_cloud = self.sorted_z_pc
            extracted_sorted_pc = self.extracted_sorted_z_pc
         

        first_index = bisect.bisect_left(extracted_sorted_pc, lowest_range)
        last_index = bisect.bisect_right(extracted_sorted_pc, highest_range)

        return sorted_point_cloud[first_index:last_index]


    def get_local_average(self, **kwargs) -> np.ndarray:
        
        x_start = kwargs['x_start']
        y_start = kwargs['y_start']
        z_start = kwargs['z_start']

        x_end = kwargs['x_end']
        y_end = kwargs['y_end']
        z_end = kwargs['z_end']

        point_cloud = self.point_cloud
        r_list = []
        g_list = []
        b_list = []

        for i, point in enumerate(point_cloud[:, :3]):

            if (point[0] <= x_start) or (point[0] >= x_end):
                continue

            if (point[1] <= y_start) or (point[1] >= y_end):
                continue
            
            if (point[2] <= z_start) or (point[2] >= z_end):
                continue

            r_list.append(point_cloud[i, 3])
            g_list.append(point_cloud[i, 4])
            b_list.append(point_cloud[i, 5])

            del point

        if len(r_list) == 0:
            return -1
        
        else:
            return np.mean(r_list), np.mean(g_list), np.mean(b_list)

    def point_intersection(self, *args):

        arr1, arr2, arr3 = args[0], args[1], args[2]
        arrays = [np.array(arr1), np.array(arr2), np.array(arr3)]
        
        def to_structured(arr):
            dtype = np.dtype([('', arr.dtype)] * arr.shape[1])
            return arr.view(dtype).flatten()
        
        structured_arrays = [to_structured(arr) for arr in arrays]
        common_structured = reduce(np.intersect1d, structured_arrays)
        
        if len(common_structured) > 0:
            return common_structured.view(arrays[0].dtype).reshape(-1, arrays[0].shape[1])
        else:
            return np.array([[-1.0] * 6])


    def get_local_average_faster(self, start_stop_list):

        x_start = start_stop_list['x_start']
        y_start = start_stop_list['y_start']
        z_start = start_stop_list['z_start']

        x_end = start_stop_list['x_end']
        y_end = start_stop_list['y_end']
        z_end = start_stop_list['z_end']

        points_in_x_range = self.get_points_in_range(
            dimension_key='x',
            start=x_start,
            end=x_end
        )

        points_in_y_range = self.get_points_in_range(
            dimension_key='y',
            start=y_start,
            end=y_end
        )

        points_in_z_range = self.get_points_in_range(
            dimension_key='z',
            start=z_start,
            end=z_end
        )
 
        amount_in_x,_ = points_in_x_range.shape
        amount_in_y,_ = points_in_y_range.shape
        amount_in_z,_ = points_in_z_range.shape
        total_possible_points = amount_in_x + amount_in_y + amount_in_z

        if(total_possible_points == 0):
            return np.array([-1.0] * 3)

        common_points = self.point_intersection(
            points_in_x_range, 
            points_in_y_range, 
            points_in_z_range
        )

        common_colors = common_points[:, 3:]
        average_color = np.mean(common_colors, axis=0)

        return np.array(average_color)

    def get_local_average_fastest(self, start_stop_dict):

        x_start = start_stop_dict['x_start']
        y_start = start_stop_dict['y_start']
        z_start = start_stop_dict['z_start']

        x_end = start_stop_dict['x_end']
        y_end = start_stop_dict['y_end']
        z_end = start_stop_dict['z_end']

        mask = ((self.point_cloud[:, 0] >= x_start) & (self.point_cloud[:, 0] < x_end) &
                (self.point_cloud[:, 1] >= y_start) & (self.point_cloud[:, 1] < y_end) &
                (self.point_cloud[:, 2] >= z_start) & (self.point_cloud[:, 2] < z_end))
        
        points_in_voxel = self.point_cloud[mask]
        
        if len(points_in_voxel) == 0:
            return np.array([-1.0] * 3)
        
        return np.mean(points_in_voxel[:, 3:6], axis=0)


    def converter(self):
        
        voxelMatrix = np.empty((self.x_size, self.y_size, self.z_size), dtype='object')

        for i in range(self.x_size):

            x_start = i/self.x_size
            x_end = (i+1)/self.x_size

            for j in range(self.y_size):

                y_start = j/self.y_size
                y_end = (j+1)/self.y_size

                for k in range(self.z_size):

                    z_start = k/self.z_size
                    z_end = (k+1)/self.z_size

                    pixel = self.get_local_average(
                        x_start=x_start,
                        y_start=y_start,
                        z_start=z_start,
                        x_end=x_end,
                        y_end=y_end,
                        z_end=z_end
                    )
                    if pixel != -1:
                        voxelMatrix[i][j][k] = pixel
        
        return voxelMatrix

    def set_pixel(self, **kwargs):

        x_start = kwargs['x_start']
        y_start = kwargs['y_start']
        z_start = kwargs['z_start']

        x_end = kwargs['x_end']
        y_end = kwargs['y_end']
        z_end = kwargs['z_end']

        i = kwargs["i"]
        j = kwargs["j"]
        k = kwargs["k"]

        pixel = self.get_local_average_faster(
            x_start=x_start,
            y_start=y_start,
            z_start=z_start,
            x_end=x_end,
            y_end=y_end,
            z_end=z_end
        )

        if not (pixel == -1).any():
            self.model[i][j][k] = pixel

    def check_time(self, **kwargs):

        percentage_complete = kwargs["percentage_complete"]
        start_time = kwargs["start_time"]
        current_time = time.perf_counter() - start_time
        total_time = current_time / percentage_complete
        remaining_time = total_time - current_time
        print(remaining_time)

def save_data(directory, model):

    np.save(directory, model)

def load_data(directory) -> np.ndarray:

    model = np.load(directory, allow_pickle=True)
    return model

def convert_to_point_cloud(model):

    x_size, y_size, z_size,_ = model.shape

    x_offset = 0.5 / x_size
    y_offset = 0.5 / y_size
    z_offset = 0.5 / z_size
    point_cloud = np.array([])

    for i in range(x_size):

        for j in range(y_size):

            for k in range(z_size):
                
                if (model[i,j,k] != -1).any():

                    x_pos = (i+x_offset)/4
                    y_pos = (j+y_offset)/4
                    z_pos = (k+z_offset)/4
                    point = np.concatenate([[x_pos, y_pos, z_pos], model[i,j,k]],
                        dtype="float16")
                    point_cloud = np.append(point_cloud, point)
    
    return point_cloud.reshape(-1, 6)

def converter_to_voxel(point_cloud, x_size, y_size, z_size):

    voxel_matrix = VoxelMatrix(
        point_cloud=point_cloud,
        x_size=x_size,
        y_size=y_size,
        z_size=z_size
    ) 

    voxel_model = np.empty((x_size, y_size, z_size), dtype='object')

    x_factor = 1.0 / x_size
    y_factor = 1.0 / y_size
    z_factor = 1.0 / z_size


    worker_items = [
    {
        "x_start": i * x_factor, "x_end": (i + 1) * x_factor,
        "y_start": j * y_factor, "y_end": (j + 1) * y_factor,
        "z_start": k * z_factor, "z_end": (k + 1) * z_factor,
        "i": i, "j": j, "k": k
    }
    for i in range(x_size)
    for j in range(y_size)
    for k in range(z_size)
    ]
    

    '''
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = np.array(list(executor.map(voxel_matrix.get_local_average_fastest, worker_items)))
    '''


    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(voxel_matrix.get_local_average_fastest, item) for item in worker_items]
        results = []
        for future in tqdm(futures, desc="Processing"):
            results.append(future.result())
        results = np.array(results)
       
    return results.reshape(x_size, y_size, z_size, 3)

