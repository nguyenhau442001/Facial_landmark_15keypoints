# INTRODUCTION 
Link dataset:https://www.kaggle.com/datasets/parthgupta28/facial-keypoints-detection

I was implemented project facial landmarks (15keypoints) with dataset from Kaggle. Datas file include 2 major part: training.csv (7049,31) and test.csv (1783,9216).
![image](https://user-images.githubusercontent.com/105471622/174613775-b86e098e-02d1-42a1-9a03-4bc9b1b84afc.png).

But files have several missing elements, so I have fullfill by using method='ffill'.

Display several training images

![image](https://user-images.githubusercontent.com/105471622/174614196-25c1d5eb-b44e-4625-9bc6-993346ec4c9a.png)
![image](https://user-images.githubusercontent.com/105471622/174614653-df64a642-b113-4797-b4c1-ac0730286cf7.png)

Because the data is limited, so we can augment the image by rotate image.

![image](https://user-images.githubusercontent.com/105471622/174615018-608b8efe-8b05-49f6-a0a0-429b8b631d31.png)

Display rotated image.

![image](https://user-images.githubusercontent.com/105471622/174615125-a5e29922-7a50-4fd7-8511-008692059739.png)

Next, I have to build model using Convolutional Neural Network. 
![image](https://user-images.githubusercontent.com/105471622/174615404-12d1a117-c9b3-4555-870b-382d504feed9.png)

![image](https://user-images.githubusercontent.com/105471622/174615559-6511cdb3-0d33-4aa0-8511-2910a1405d45.png)



