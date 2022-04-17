import pandas as pd  # 做数据分析
import numpy as np  # 数据处理
import matplotlib.pyplot as plt  # 绘图

file_path1 = "D:\\VSC prj\\cnn\\dataFile_1.0_80.txt"
f = open(file_path1, 'r')
file = f.read()
f.close()
a = file.split()
a = np.array(a,dtype=float)
# print(a.shape) 8730
a = a.reshape(873,10)

file_path2 = "C:\\Users\\wydx\\Desktop\\噪声数据_最新版\\data2\\1_1.0_80_noise1.xlsx"
df = pd.read_excel(file_path2, sheet_name="Sheet1")  # sheet_name不指定时默认返回全表数据
b = np.array(df)

predict = []
for i in range(2, 873):
    predict.append(a[i, :])
predict_test = np.array(predict)

true = []
for i in range(2, 873):
    true.append(b[i, :])
true = np.array(true)

for i in range(0, 10):
    plt.plot(true[:, i], c='red')
    plt.plot(predict_test[:, i], c='blue')
    plt.show()
