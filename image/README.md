# PoseNet：第六届研究生智慧城市挑战赛基于全景图像的相机拍摄位置估计
## Requirement
1.tensrflow==1.2
2.python == 2.7
3.opencv-python
4.tqdm
5.cuda8.0 cudnn5.1
## File structure
1.data_argument.py 由于每个场景下提供的全景图像只有50-70张，这对于训练一个鲁棒性强的模型是远远不够的，因此采用了包括mix_up的多种数据增强方式，进行数据扩充；
2.compute_mean.py 由于相机拍摄位置 （xyz）差距较大，因此首先对标签进行归一化处理，并且为了便于学习，对归一化后的数据进行了放大处理；
3 posenet.py 基于原始posenet改进后的模型
4.network.py posenet中用的一些层
5.train.py 基于比赛数据集的每个场景dataset训练出一个模型，并将模型保存在model_dir
6.test_new.py 测试代码
## Innovation
1.gloab average pooling
2.panoramic image block input
3.Generate a virtual sample
## Reference