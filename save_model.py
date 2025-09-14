import model_dataset as Dataset


flower_dataset = Dataset.PointCloud(default_path="Point_Dataset/", size_of_dataset=103)
flower_dataset.save_data('normalized_dataset.npy')