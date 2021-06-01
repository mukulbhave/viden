#Vehicle Identification(Viden)

The goal is to create a service to detect fraud in license plate usage by way of 
1. Identifying vehicle  license plate number.
2. Vehicle make, model 
3. Use vehicle number from step 1 to fetch vehicle details from govt vehicle database 
   using Vahan e-services and flag any discrepancy found.
   ![Evalauation Flow](img/flow.svg)                 
Currently we have created deep learning model to detect and recognise car license plate number only due to resource constraints.Even just for cars our tarining dataset was comparitively small (~300) car images downloaded from internet. For text detection model we used a synthetic dataset. The dataset size was ~50k images which is comparitively smaller.CRNN+CTC model was trained on 8Gib GPU. The training ,validation and test set were in 80:10:10 ratio in both models training. 

For YOLOv2 training we started with the pretrained weights of the Darknet and fine-tuned the model by changing the weights of the last two layers. The model is trained for 30 epochs used the learning rate , weight decay and momentum  same as used for training the YOLOv2 Darknet-19 on both COCO and VOC dataset.

YOLOv2 Training and Validation Loss for Car License Plate Detection
![Training Loss](img/loss.svg)                  ![Validation Loss](img/val_loss.svg)


CNN+RNN+CTC Training and Validation Loss For License Number Recognition

![Training Loss](img/loss_CRNN.svg)                  ![Validation Loss](img/val_loss_CRNN.svg)

##Demo App

[The Demo app is hosted on Google Cloud Run]:( https://viden-n6rafmjufa-el.a.run.app)

### Sample Input Output
####Input Image
![Car Image](test_sample_img/1_ (5).jpg)
####Output 
![Car License Number ](img/Sample_output.MHT)

##Tech Stack
1. Html/jquery
2. Python , TensorFlow and Keras,Flask
3. YOLOv2 YAD2K model for license plate detection
4. CNN+RNN+CTC model to recognize license number from license plate.
5. Docker and GCP for deployment


Implementation References


YOLOv2 YAD2K refrence https://github.com/allanzelener/YAD2K

CRNN+CTC Model https://github.com/TheAILearner/A-CRNN-model-for-Text-Recognition-in-Keras