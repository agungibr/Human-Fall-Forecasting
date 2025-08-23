# HumanFallForecasting

![Project Logo](https://github.com/AndiDemon/HumanFallForecasting/blob/main/Figures/prediction_sequence.jpg)

**Dataset**

https://drive.google.com/drive/folders/1GCpfPpiiZuvNxmQ1_SEApRYEEYiW41rz?usp=share_link

**Overview**

TelUP Human Fall Motion Dataset is an open-source research development to forecast and detect the human fall motion and analyze the motion using physical parameters and deep learning model. The dataset is collected under controlled condition in an indoor environment. Under this challenge, we provide the pose feature of single human body in the frame preprocessed with YOLOv11 Pose model. With this data, the participants are asked to put the effort to make the best model to forecast and classify the fall human motion under certain specific variables.

**Dataset**

The TelUP Dataset is available on our Google Drive. The pose information of TelUP Dataset is extracted using YOLOv11 to obtain 17 keypoints per frame. These keypoints are normalized and organized into sequences using a sliding window technique. The resulting sequences are then split into training and testing sets based on subject. 
Finally, the data is used to train and evaluate multitask deep learning models for simultaneous forecasting and classification. 

**Citation**
