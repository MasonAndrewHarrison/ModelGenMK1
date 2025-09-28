import open3d as o3d

def show_model(point_cloud):

    

    if point_cloud is None:
        raise ValueError("Array is empty")

    else:
        pcd_list = o3d.geometry.PointCloud()
        pcd_list.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
        pcd_list.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:])

        #o3d.visualization.draw_geometries([pcd_list])

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd_list)
    
        # Get render option and set point size
        render_option = vis.get_render_option()
        render_option.point_size = 15.0  # Increase this value to make points bigger
    
        vis.run()
        vis.destroy_window()

