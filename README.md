Vehicle Identification(Viden)

The goal is to create a service to detect fraud in license plate usage by way of 
1. Identifying vehicle  license plate number.
2. Vehicle make, model 
3. Use vehicle number from step 1 to fetch vehicle details from govt vehicle database 
   using Vahan e-services and flag any discrepancy found.
   
Currently we have created deep learning model to detect and recognise license plate number. Due to resource constraints our tarining dataset was comparitively small (~500) car images comprising images downloaded from internet primarily. For text detection model we used a synthetic dataset. The dataset size was ~200k images. The training ,validation and test set were in 80:10:10 ratio in both cases.



The Demo app is hosted on GCP URL: https://viden-n6rafmjufa-el.a.run.app

Input Image

Output:


Tech Stack
1. Html/jquery
2. Python , TensorFlow and Keras,Flask
3. YOLOv2 YAD2K model for license plate detection
4. CNN+RNN+CTC model to recognize license number from license plate.
5. Docker and GCP for deployment


Implementation References


YOLOv2 YAD2K refrence https://github.com/allanzelener/YAD2K

CRNN+CTC Model https://github.com/TheAILearner/A-CRNN-model-for-Text-Recognition-in-Keras