detection中的train.xlsx为筛选的训练集合的代码，经过了手工处理之后得到的训练集合，test.xlsx为筛选的测试集合的代码，经过了手工处理之后得到的测试集合，输入的是特征提取18个通道进行了DWT变换后的特征值。
先进行预处理(preprocessi.py)，在进行特征提取(feature.py)，将我们提取到的特征值按照chb-mit给出的癫痫时间进行标记，最后放入(detection.py)检测。
