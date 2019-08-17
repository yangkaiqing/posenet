# posenet
# PoseNet：第六届研究生智慧城市挑战赛基于全景图像的相机拍摄位置估计  
## Requirement  
1.tensrflow==1.2  
2.python == 2.7  
3.opencv-python  
4.tqdm  
5.cuda8.0 cudnn5.1  
## File structure  
1.data_argument.py Since there are only 50-70 panoramic images provided in each scene, it is not enough to train a robust model. Therefore, a variety of data enhancement methods including mix_up are adopted for data expansion.  
2.compute_mean.py Since the camera shooting position (xyz) is large, the label is first normalized, and the normalized data is enlarged for ease of learning;  
3 posenet.py Improved model based on original posenet  
4.network.py posenet Some layers used in posenet  
5.train.py Train a model based on each scene dataset of the game dataset and save the model in model_dir  
6.test_new.py test code  
## Innovation
1.gloab average pooling  
![gloab_average_pooling_improve](https://github.com/yangkaiqing/posenet/blob/master/image/global%20average%20pooling.png)  
2.panoramic image block input  
![model1](https://github.com/yangkaiqing/posenet/blob/master/image/model.png)  
3.Generate a virtual sample  
## Result
1.result for data argumentation  
![reult1](https://github.com/yangkaiqing/posenet/blob/master/image/result_for_dataargumention.png)  
2.result for model structure change  
![reult2](https://github.com/yangkaiqing/posenet/blob/master/image/result_for_gloabaveragepooling.png)  
## Reference
1.Baseline model as described in the ICCV 2015 paper PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization Alex Kendall, Matthew Grimes and Roberto Cipolla [http://mi.eng.cam.ac.uk/projects/relocalisation/]  
2.https://github.com/kentsommer/tensorflow-posenet  
3.https://github.com/SoftwareGift/FeatherNets_Face-Anti-spoofing-Attack-Detection-Challenge-CVPR2019  
