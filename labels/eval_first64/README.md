Labels for the eval recording (first 64 images, Roboflow export, YOLO format), Rafal.
Roboflow: rafals-workspace-rnj5i/fire_extinguishers_label v1, no preprocessing, no augmentation.

Class IDs: 0 = decoy, 1 = real (see data.yaml)

Each .txt has one line per box: class x_center y_center width height (normalized 0-1).
Fix after export: camera_color_image_1727164272332739264 - small box on the pillar changed real -> decoy (it is the printed copy).
