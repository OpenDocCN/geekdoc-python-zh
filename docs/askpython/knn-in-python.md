# Python 中的 KNN——简单实用的实现

> 原文：<https://www.askpython.com/python/examples/knn-in-python>

读者朋友们，你们好！在这篇文章中，我们将关注对 Python 中 KNN 的**理解和实现。**

所以，让我们开始吧！！

* * *

## 什么是 KNN 算法？

KNN 是 K 近邻的首字母缩写。它是一种有监督的机器学习算法。KNN 基本上用于分类和回归。

KNN 不假设任何底层参数，即它是一个`non-parametric`算法。

* * *

### KNN 算法遵循的步骤

*   它最初将训练数据存储到环境中。
*   当我们提出用于预测的数据时，Knn 根据训练数据集为新的测试记录选择 **k 个最相似/相似的数据值**。
*   此外，使用`Euclidean or Manhattan distance`为新测试点选择 k 个最相似的邻居。基本上，他们计算测试点和训练数据值之间的距离，然后选择 K 个最近的邻居。
*   最后，将测试数据值分配给包含测试数据的 K 个最近邻的最大点的类或组。

* * *

### K-NN 的真实例子

**问题陈述—**考虑一袋珠子(训练数据)，有两种颜色——绿色和蓝色。

所以，这里有两类:绿色和蓝色。我们的任务是找到一个新的珠子“Z”会落在哪个类中。

**解决方案—**最初，我们随机选择 K 的值。现在假设 K=4。因此，KNN 将使用所有训练数据值(一袋珠子)计算 Z 的距离。

此外，我们选择最接近 Z 的 4(K)个值，然后尝试分析 4 个邻居中的大多数属于哪个类。

最后，Z 被分配一类空间中的大多数邻居。

* * *

## KNN 在 Python 中的实现

现在，让我们试着用 KNN 的概念来解决下面的回归问题。

我们得到了一个数据集，其中包含了根据各种环境条件选择租赁自行车的人数的历史数据。

你可以在这里找到数据集[。](https://github.com/Safa1615/BIKE-RENTAL-COUNT/blob/master/day.csv)

所以，让我们开始吧！

* * *

### 1.加载数据集

我们已经利用 [Pandas 模块](https://www.askpython.com/python-modules/pandas/python-pandas-module-tutorial)将数据集加载到使用`pandas.read_csv()`函数的环境中。

```py
import pandas 
BIKE = pandas.read_csv("Bike.csv")

```

### 2.选择正确的功能

我们利用[相关回归分析](https://www.askpython.com/python/examples/correlation-matrix-in-python)技术从数据集中选择重要变量。

```py
corr_matrix = BIKE.loc[:,numeric_col].corr()
print(corr_matrix)

```

**相关矩阵**

```py
               temp     atemp       hum  windspeed
temp       1.000000  0.991738  0.114191  -0.140169
atemp      0.991738  1.000000  0.126587  -0.166038
hum        0.114191  0.126587  1.000000  -0.204496
windspeed -0.140169 -0.166038 -0.204496   1.000000

```

由于“temp”和“atemp”高度相关，我们从数据集中删除了“atemp”。

```py
BIKE = BIKE.drop(['atemp'],axis=1)

```

### 3.分割数据集

我们已经利用 [train_test_split()函数](https://www.askpython.com/python/examples/split-data-training-and-testing-set)将数据集分成 80%的训练数据集和 20%的测试数据集。

```py
#Separating the dependent and independent data variables into two data frames.
from sklearn.model_selection import train_test_split 

X = bike.drop(['cnt'],axis=1) 
Y = bike['cnt']

# Splitting the dataset into 80% training data and 20% testing data.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.20, random_state=0)

```

### 4.定义误差指标

由于这是一个回归问题，我们将 [MAPE](https://www.askpython.com/python/examples/mape-mean-absolute-percentage-error?_thumbnail_id=9324) 定义为如下所示的误差指标

```py
import numpy as np
def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return Mape

```

### 5.建立模型

`sklearn.neighbors module`包含实现 Knn 的`KNeighborsRegressor()`方法，如下所示

```py
#Building the KNN Model on our dataset
from sklearn.neighbors import KNeighborsRegressor
KNN_model = KNeighborsRegressor(n_neighbors=3).fit(X_train,Y_train)

```

此外，我们使用 [predict()函数](https://www.askpython.com/python/examples/python-predict-function)来预测测试数据。

```py
KNN_predict = KNN_model.predict(X_test) #Predictions on Testing data

```

### 6.准确性检查！

我们调用上面定义的 MAPE 函数来检查分类错误并判断模型预测的准确性。

```py
# Using MAPE error metrics to check for the error rate and accuracy level
KNN_MAPE = MAPE(Y_test,KNN_predict)
Accuracy_KNN = 100 - KNN_MAPE
print("MAPE: ",KNN_MAPE)
print('Accuracy of KNN model: {:0.2f}%.'.format(Accuracy_KNN))

```

**Knn 的精度评估—**

```py
MAPE:  17.443668778014253
Accuracy of KNN model: 82.56%.

```

* * *

## 结论

到此，我们就结束了这个话题。如果你遇到任何问题，欢迎在下面评论。

更多与 Python 相关的帖子，敬请关注，在此之前，祝你学习愉快！！🙂