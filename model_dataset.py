import open3d as o3d
from plyfile import PlyData
import numpy as np
from typing import overload
import bisect
import pandas as pd
import time
from functools import reduce

class Visualizer: 

    @staticmethod
    def show_model(point_cloud):

        if point_cloud is None:
            raise ValueError("Array is empty")

        else:
            pcd_list = o3d.geometry.PointCloud()
            pcd_list.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
            pcd_list.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:])

            o3d.visualization.draw_geometries([pcd_list])


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

            self.model = self.faster_converter()
        
        elif 'directory' in kwargs:
            self.directory = kwargs['directory']
            self.model = self.load_data()
            self.x_size, self.y_size, self.z_size = self.model.shape

    def get_model(self):
        return self.model

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
        
        # Convert each array to structured array for faster comparison
        def to_structured(arr):
            dtype = np.dtype([('', arr.dtype)] * arr.shape[1])
            return arr.view(dtype).flatten()
        
        structured_arrays = [to_structured(arr) for arr in arrays]
        
        # Find intersection of all arrays
        common_structured = reduce(np.intersect1d, structured_arrays)
        
        # Convert back to regular array
        if len(common_structured) > 0:
            return common_structured.view(arrays[0].dtype).reshape(-1, arrays[0].shape[1])
        else:
            return np.array([[-1.0] * 6])
            #return np.array([]).reshape(0, arrays[0].shape[1])


    def get_local_average_faster(self, **kwargs):

        x_start = kwargs['x_start']
        y_start = kwargs['y_start']
        z_start = kwargs['z_start']

        x_end = kwargs['x_end']
        y_end = kwargs['y_end']
        z_end = kwargs['z_end']

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

    def faster_converter(self):

        voxelMatrix = np.empty((self.x_size, self.y_size, self.z_size), dtype='object')

        x_factor = 1.0 / self.x_size
        y_factor = 1.0 / self.y_size
        z_factor = 1.0 / self.z_size

        total_pixel_amount = self.x_size * self.y_size * self.z_size
        counter = 0
        loading_timer = time.perf_counter() 

        for i in range(self.x_size):

            x_start = i * x_factor
            x_end = x_start + x_factor

            for j in range(self.y_size):

                y_start = j * y_factor
                y_end = y_start + y_factor

                for k in range(self.z_size):

                    z_start = k * z_factor
                    z_end = z_start + z_factor

                    start = time.perf_counter()
                    pixel = self.get_local_average_faster(
                        x_start=x_start,
                        y_start=y_start,
                        z_start=z_start,
                        x_end=x_end,
                        y_end=y_end,
                        z_end=z_end
                    )
                    end = time.perf_counter()

                    counter += 1
                    percentage_complete = counter / total_pixel_amount
                    current_time = end - loading_timer
                    total_time = current_time / percentage_complete
                    remaining_time = total_time - current_time
                    print(f"Exectution time: {end - start:.4f} seconds {(percentage_complete):.8f}% | min remaining {remaining_time/60:.4f} | total estimated time {(total_time/60):.4f}")

                    if not (pixel == -1).any():
                        voxelMatrix[i][j][k] = pixel
        
        return voxelMatrix       

    def save_data(self):
        
        np.save(self.directory, self.model)

    def load_data(self) -> np.ndarray:

        model = np.load(self.directory, allow_pickle=True)
        return model

    def convert_to_point_cloud(self):

        x_offset = 0.5 / self.x_size
        y_offset = 0.5 / self.y_size
        z_offset = 0.5 / self.z_size

        for i in range(self.x_size):

            for j in range(self.y_size):

                for k in range(self.z_size):

                    x_pos = (i+x_offset)/4
                    pixel = x_pos, self.model[i,j,k] 
                    print(pixel)




