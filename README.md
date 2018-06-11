# 使用CNN、RNN识别微博登录验证码
- 本项目来源于微博数据获取软件的部分源码，最近给本科生做爬虫讲座，有同学需要，故整理上传。
- 本项目分别使用卷积神经网络VGG、GoogLeNet、ResNet，循环神经网络LSTM，attention机制进行验证码识别。
  - 使用架构 
  1. VGG
  2. GoogLeNet
  3. ResNet
  4. CNN + LSTM
  5. CNN + LSTM + CTC loss
  6. CNN + LSTM + attention
- 语言：Python 3.6
- 依赖包：Keras, Tensorflow, numpy, matplotlib, opencv-python, imutils, scikit-learn

## 微博验证码形式
![image](http://github.com/xukunxkxk/WeiboCaptchaRecognize/tree/master/model/6pAVy.jpg)
