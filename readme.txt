C:\git\viden\python_code\yolo>

1. Create dataset . 
Use labelImg program to create bounding boxes and classes. labelImg\data\predefined_classes.txt needs to be modified to reflect the use case

2. Create a folder say "vehicle_data".This will be root to your data. Then create a folder "data" inside this. Put all images and voc xml inside this.

3. Put predefined_classes.txt into "vehicle_data" folder

4. run python process.py -d <path to data folder> . This will create a distribution of train.txt and test.txt sets. We need to pramaterize the percent as well.

5. run python voc_to_hdf5.py -d <path to "vehicle_data">. This script will create the dataset file .hdf5.

6. run python retrain_yolo.py -d <path to "vehicle_data" .hdf5 file> -c <path to predefined_classes.txt file>