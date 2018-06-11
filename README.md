# 使用CNN、RNN识别微博登录验证码
- 本项目来源于微博数据获取软件的部分源码，最近给本科生做爬虫讲座，有同学需要，故整理上传。
- 本项目分别使用卷积神经网络VGG、GoogLeNet、ResNet，循环神经网络LSTM，attention机制进行验证码识别。
- 使用架构 
  1. VGG
  2. GoogLeNet
  3. ResNet
  4. CNN + LSTM
  5. CNN + LSTM + attention
- 语言：Python 3.6
- 依赖包：Keras, Tensorflow, numpy, matplotlib, opencv-python, imutils, scikit-learn

## 项目文件
- config
  - config.py 存储训练配置文件，如验证码字符数、图片长度宽度等。
- loss_accuracy 存储训练loss与accuracy图像，以便调参
- model 存储训练模型
- network
  - cnn
    - googlenet.py GoogLeNet的CNN结构
    - head.py softmax分类器
    - resnet.py ResNet的CNN结构
    - vgg.py VGG的CNN结构
  - rnn
    - attention.py Keras Attention layer的实现
    - lstm.py CNN+LSTM，CNN+LSTM+Attention
- train
   - train.py 训练文件
   - predict.py 预测文件
- utils
  - callback.py loss、accuracy计算，模型存储
  - generator.py 训练batch数据生成
  - read_features.py 验证码图片读取与预处理
## 模型训练与预测
1. config.py里定义配置
  - IMAGES_PATH 验证码图片的位置，文件名为验证码的Ground Truth
  - CLASSES_LIST 验证码全集, 如[0, 1, 2, 3, 4, 5, a, b, c]
  - CHAR_NUMBERS 验证码所包含的字符数
  - MODEL_PATH 训练模型所储存的位置
  - OUTPUT_PATH 训练loss与accuracy所处的位置
  - HEIGHT 图片的高度
  - WIDTH 图片的宽度
  - DEPTH 图片的深度
  - EPOCH 训练轮数
  - INIT_LR 初始学习率
  - BATCH_SIZE 训练Batch大小
  - NET 所使用的训练网络类型 已实现 {'VGG', 'ResNet', 'GoogLeNet', 'LSTM', 'AttentionLSTM'}
  - SPLIT 是否分开label训练 当使用LSTM时，这个参数必须设为True
  - CNN_MODEL 预测所使用的模型
2. 训练
  定义好上述配置后就可以训练了，运行train.py文件
3. 预测
  向predict.py传入带预测图片地址，程序会打印所识别的验证码
  python3 predict.py -p image_path



## 微博验证码形式
![image](https://github.com/xukunxkxk/WeiboCaptchaRecognize/raw/master/model/6pAVy.jpg)  
长：100px, 宽：40px，字符数：5，由大小写字母+数字组成。  
这样的验证码有一定的扭曲、粘连、干扰线，使用传统的分割+识别效果可能受到影响。  
本项目不分割图片，采用端到端的形式进行识别。所考虑的模型主要是基于CNN提取图片特征，直接进行分类或是使用RNN进行序列建模。

## CNN识别验证码
使用卷积神经网络提取图片特征，多个全连接层分别识别每一位的验证码。  
以VGG网络为例，ResNet， GoogLeNet类似
卷积 kernel size 3*3 filter [32, 64, 128, 256, 512] * 2  
每两个卷积之间加入MaxPool与dropout
所使用的类似VGG的结构可视化如下：  
![image](https://github.com/xukunxkxk/WeiboCaptchaRecognize/raw/master/model/VGG.png)  
在VGG网络提取图片特征之后，送入五个softmax层进行分类，之后将这些五个向量合并，一起计算交叉熵作为损失函数。  
与传统的VGG不同的是，在设计网络的时候，1.在卷积后连接了BN层，加快收敛， 2.采取了GlobalEveryPooling替代全连接层，减少参数  

## RNN识别验证码
使用卷积神经网络提取图片特征，RNN进行序列建模，识别每一位验证码  
以AttentionLSTM为例：  
![image](https://github.com/xukunxkxk/WeiboCaptchaRecognize/raw/master/model/AttentionLSTM.png)  
模型CNN部分与VGG的类似，CNN 部分的输出为m * n * filter_size的特征，m为图片的长度，n为图片的宽度.  
将图片的“每一条”作为特征，输入LSTM，即m * n * filter_size 转变为  n *(m*filter_size)，  
在LSTM之上使用Attention、连接Softmax进行分类。  

## 性能
在训练集为4w，验证集为1w的数据集上，各模型在验证机上的表现如下：   
VGG: 93.64%  
GoogLeNet: 93.28%
ResNet: 96.41%
VGG+LSTM: 93.58%
VGG+LSTM+Attention: 95.64%
