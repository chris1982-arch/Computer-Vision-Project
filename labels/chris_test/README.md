Labels for the test recording (88 images, Roboflow export, YOLO format).

Class IDs: 0 = decoy, 1 = real

Each .txt has one line per box: class x_center y_center width height (normalized 0-1).
File names end with ".rf.<hash>" (added by Roboflow); the image name is the part before ".rf.".
