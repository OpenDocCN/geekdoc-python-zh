### 导航

+   索引

+   上一页 |

+   Python 多元分析小册子 0.1 文档 »

# Python 多元分析小册子

本小册子告诉您如何使用 Python 生态系统进行一些简单的多元分析，重点放在主成分分析（PCA）和线性判别分析（LDA）上。

本小册子假定读者对多元分析有一些基本知识，小册子的主要重点不是解释多元分析，而是解释如何使用 Python 进行这些分析。

如果您是多元分析的新手，并且想要了解这里介绍的任何概念更多信息，有许多良好的资源，例如 Hair 等人的《多元数据分析》或 Everitt 和 Dunn 的《应用多元数据分析》。

在本小册子的示例中，我将使用来自 [UCI 机器学习库](http://archive.ics.uci.edu/ml) [http://archive.ics.uci.edu/ml] 的数据集。

## 设置 Python 环境

### 安装 Python

尽管有多种方法可以将 Python 安装到您的系统中，但为了无忧安装和快速启动，我强烈建议下载并安装 [Anaconda](https://www.continuum.io/downloads) [https://www.continuum.io/downloads]，这是由 [Continuum](https://www.continuum.io) [https://www.continuum.io] 提供的 Python 发行版，包含核心包以及大量用于科学计算的包和工具，可以轻松更新它们，安装新包，创建虚拟环境，并提供诸如这个 [Jupyter notebook](https://jupyter.org) [https://jupyter.org]（以前称为 ipython notebook）之类的 IDE。

本笔记本是使用 Python 2.7 版本创建的。有关详细信息，包括其他库的版本，请参见下面的 `%watermark` 指令。  ### 库

[Python](https://en.wikipedia.org/wiki/Python_%28programming_language%29) [https://en.wikipedia.org/wiki/Python_%28programming_language%29] 通常比其他语言的开箱即用功能少，这是因为它是一种通用编程语言，采用更模块化的方法，依赖其他包来执行专门的任务。

这里使用以下库：

+   [pandas](http://pandas.pydata.org) [http://pandas.pydata.org]: 用于存储数据框和操作的 Python 数据分析库。

+   [numpy](http://www.numpy.org) [http://www.numpy.org]: Python 科学计算库。

+   [matplotlib](http://matplotlib.org) [http://matplotlib.org]: Python 绘图库。

+   [seaborn](http://stanford.edu/~mwaskom/software/seaborn/) [http://stanford.edu/~mwaskom/software/seaborn/]: 基于 matplotlib 的统计数据可视化。

+   [scikit-learn](http://scikit-learn.org/stable/) [http://scikit-learn.org/stable/]: Sklearn 是一个用于 Python 的机器学习库。

+   [scipy.stats](http://docs.scipy.org/doc/scipy/reference/stats.html) [http://docs.scipy.org/doc/scipy/reference/stats.html]: 提供了许多概率分布和统计函数。

如果您已经安装了 Anaconda Python 发行版，则应该已经为您安装了这些。

库的版本如下：

```
from __future__ import print_function, division  # for compatibility with python 3.x
import warnings
warnings.filterwarnings('ignore')  # don't print out warnings

%install_ext https://raw.githubusercontent.com/rasbt/watermark/master/watermark.py
%load_ext watermark
%watermark -v -m -p python,pandas,numpy,matplotlib,seaborn,scikit-learn,scipy -g

```

```
Installed watermark.py. To use it, type:
  %load_ext watermark
CPython 2.7.11
IPython 4.0.3

python 2.7.11
pandas 0.17.1
numpy 1.10.4
matplotlib 1.5.1
seaborn 0.7.0
scikit-learn 0.17
scipy 0.17.0

compiler   : GCC 4.2.1 (Apple Inc. build 5577)
system     : Darwin
release    : 13.4.0
machine    : x86_64
processor  : i386
CPU cores  : 4
interpreter: 64bit
Git hash   : b584574b9a5080bac2e592d4432f9c17c1845c18

```  ### 导入库

```
from pydoc import help  # can type in the python console `help(name of function)` to get the documentation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy import stats
from IPython.display import display, HTML

# figures inline in notebook
%matplotlib inline

np.set_printoptions(suppress=True)

DISPLAY_MAX_ROWS = 20  # number of max rows to print for a DataFrame
pd.set_option('display.max_rows', DISPLAY_MAX_ROWS)

```  ### Python 控制台

用于快速实验和数据可视化的便笺旁边有一个附加的 Python 控制台是一个有用的工具。如果您希望拥有一个，请取消以下行的注释。

```
# %qtconsole

```  ## 将多元分析数据读入 Python

要分析多元数据，您首先需要做的事情是将其读入 Python，并绘制数据。对于数据分析，我将使用[Python 数据分析库](http://pandas.pydata.org) [http://pandas.pydata.org]（pandas，导入为`pd`），它提供了许多有用的函数用于读取和分析数据，以及一个`DataFrame`存储结构，类似于其他流行的数据分析语言中的结构，例如 R。

例如，文件 http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data 包含了来自意大利同一地区种植的三种不同培育者的葡萄酒中 13 种不同化学物质的浓度数据。数据集如下所示：

```
1,14.23,1.71,2.43,15.6,127,2.8,3.06,.28,2.29,5.64,1.04,3.92,1065
1,13.2,1.78,2.14,11.2,100,2.65,2.76,.26,1.28,4.38,1.05,3.4,1050
1,13.16,2.36,2.67,18.6,101,2.8,3.24,.3,2.81,5.68,1.03,3.17,1185
1,14.37,1.95,2.5,16.8,113,3.85,3.49,.24,2.18,7.8,.86,3.45,1480
1,13.24,2.59,2.87,21,118,2.8,2.69,.39,1.82,4.32,1.04,2.93,735
...

```

每个葡萄酒样本一行。第一列包含葡萄酒样本的培育者（标记为 1、2 或 3），接下来的十三列包含该样本中 13 种不同化学物质的浓度。列之间用逗号分隔，即它是一个没有标题行的逗号分隔（csv）文件。

可以使用`read_csv()`函数将数据读入 pandas dataframe。参数`header=None`告诉函数文件开头没有标题。

```
data = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header=None)
data.columns = ["V"+str(i) for i in range(1, len(data.columns)+1)]  # rename column names to be similar to R naming convention
data.V1 = data.V1.astype(str)
X = data.loc[:, "V2":]  # independent variables data
y = data.V1  # dependednt variable data
data

```

|  | V1 | V2 | V3 | V4 | V5 | V6 | V7 | V8 | V9 | V10 | V11 | V12 | V13 | V14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 14.23 | 1.71 | 2.43 | 15.6 | 127 | 2.80 | 3.06 | 0.28 | 2.29 | 5.640000 | 1.04 | 3.92 | 1065 |
| 1 | 1 | 13.20 | 1.78 | 2.14 | 11.2 | 100 | 2.65 | 2.76 | 0.26 | 1.28 | 4.380000 | 1.05 | 3.40 | 1050 |
| 2 | 1 | 13.16 | 2.36 | 2.67 | 18.6 | 101 | 2.80 | 3.24 | 0.30 | 2.81 | 5.680000 | 1.03 | 3.17 | 1185 |
| 3 | 1 | 14.37 | 1.95 | 2.50 | 16.8 | 113 | 3.85 | 3.49 | 0.24 | 2.18 | 7.800000 | 0.86 | 3.45 | 1480 |
| 4 | 1 | 13.24 | 2.59 | 2.87 | 21.0 | 118 | 2.80 | 2.69 | 0.39 | 1.82 | 4.320000 | 1.04 | 2.93 | 735 |
| 5 | 1 | 14.20 | 1.76 | 2.45 | 15.2 | 112 | 3.27 | 3.39 | 0.34 | 1.97 | 6.750000 | 1.05 | 2.85 | 1450 |
| 6 | 1 | 14.39 | 1.87 | 2.45 | 14.6 | 96 | 2.50 | 2.52 | 0.30 | 1.98 | 5.250000 | 1.02 | 3.58 | 1290 |
| 7 | 1 | 14.06 | 2.15 | 2.61 | 17.6 | 121 | 2.60 | 2.51 | 0.31 | 1.25 | 5.050000 | 1.06 | 3.58 | 1295 |
| 8 | 1 | 14.83 | 1.64 | 2.17 | 14.0 | 97 | 2.80 | 2.98 | 0.29 | 1.98 | 5.200000 | 1.08 | 2.85 | 1045 |
| 9 | 1 | 13.86 | 1.35 | 2.27 | 16.0 | 98 | 2.98 | 3.15 | 0.22 | 1.85 | 7.220000 | 1.01 | 3.55 | 1045 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 168 | 3 | 13.58 | 2.58 | 2.69 | 24.5 | 105 | 1.55 | 0.84 | 0.39 | 1.54 | 8.660000 | 0.74 | 1.80 | 750 |
| 169 | 3 | 13.40 | 4.60 | 2.86 | 25.0 | 112 | 1.98 | 0.96 | 0.27 | 1.11 | 8.500000 | 0.67 | 1.92 | 630 |
| 170 | 3 | 12.20 | 3.03 | 2.32 | 19.0 | 96 | 1.25 | 0.49 | 0.40 | 0.73 | 5.500000 | 0.66 | 1.83 | 510 |
| 171 | 3 | 12.77 | 2.39 | 2.28 | 19.5 | 86 | 1.39 | 0.51 | 0.48 | 0.64 | 9.899999 | 0.57 | 1.63 | 470 |
| 172 | 3 | 14.16 | 2.51 | 2.48 | 20.0 | 91 | 1.68 | 0.70 | 0.44 | 1.24 | 9.700000 | 0.62 | 1.71 | 660 |
| 173 | 3 | 13.71 | 5.65 | 2.45 | 20.5 | 95 | 1.68 | 0.61 | 0.52 | 1.06 | 7.700000 | 0.64 | 1.74 | 740 |
| 174 | 3 | 13.40 | 3.91 | 2.48 | 23.0 | 102 | 1.80 | 0.75 | 0.43 | 1.41 | 7.300000 | 0.70 | 1.56 | 750 |
| 175 | 3 | 13.27 | 4.28 | 2.26 | 20.0 | 120 | 1.59 | 0.69 | 0.43 | 1.35 | 10.200000 | 0.59 | 1.56 | 835 |
| 176 | 3 | 13.17 | 2.59 | 2.37 | 20.0 | 120 | 1.65 | 0.68 | 0.53 | 1.46 | 9.300000 | 0.60 | 1.62 | 840 |
| 177 | 3 | 14.13 | 4.10 | 2.74 | 24.5 | 96 | 2.05 | 0.76 | 0.56 | 1.35 | 9.200000 | 0.61 | 1.60 | 560 |

178 行 × 14 列

在这种情况下，178 个葡萄酒样本的数据已被读入变量`data`。## 绘制多变量数据

一旦您将多变量数据集读入 python，下一步通常是绘制数据的图表。

### 矩阵散点图

用于绘制多变量数据的一种常见方法是制作*矩阵散点图*，显示每对变量相互绘制。我们可以使用`pandas.tools.plotting`包中的`scatter_matrix()`函数来实现这一点。

要使用`scatter_matrix()`函数，您需要将要包含在图中的变量作为其输入。例如，假设我们只想包括与前五种化学品浓度对应的变量。这些存储在变量`data`的列 V2-V6 中。参数`diagonal`允许我们指定是绘制直方图(`"hist"`)还是核密度估计(`"kde"`)。我们可以通过输入以下内容从变量`data`中提取这些列：

```
data.loc[:, "V2":"V6"]

```

|  | V2 | V3 | V4 | V5 | V6 |
| --- | --- | --- | --- | --- | --- |
| 0 | 14.23 | 1.71 | 2.43 | 15.6 | 127 |
| 1 | 13.20 | 1.78 | 2.14 | 11.2 | 100 |
| 2 | 13.16 | 2.36 | 2.67 | 18.6 | 101 |
| 3 | 14.37 | 1.95 | 2.50 | 16.8 | 113 |
| 4 | 13.24 | 2.59 | 2.87 | 21.0 | 118 |
| 5 | 14.20 | 1.76 | 2.45 | 15.2 | 112 |
| 6 | 14.39 | 1.87 | 2.45 | 14.6 | 96 |
| 7 | 14.06 | 2.15 | 2.61 | 17.6 | 121 |
| 8 | 14.83 | 1.64 | 2.17 | 14.0 | 97 |
| 9 | 13.86 | 1.35 | 2.27 | 16.0 | 98 |
| ... | ... | ... | ... | ... | ... |
| 168 | 13.58 | 2.58 | 2.69 | 24.5 | 105 |
| 169 | 13.40 | 4.60 | 2.86 | 25.0 | 112 |
| 170 | 12.20 | 3.03 | 2.32 | 19.0 | 96 |
| 171 | 12.77 | 2.39 | 2.28 | 19.5 | 86 |
| 172 | 14.16 | 2.51 | 2.48 | 20.0 | 91 |
| 173 | 13.71 | 5.65 | 2.45 | 20.5 | 95 |
| 174 | 13.40 | 3.91 | 2.48 | 23.0 | 102 |
| 175 | 13.27 | 4.28 | 2.26 | 20.0 | 120 |
| 176 | 13.17 | 2.59 | 2.37 | 20.0 | 120 |
| 177 | 14.13 | 4.10 | 2.74 | 24.5 | 96 |

178 行×5 列

要使用`scatter_matrix()`函数制作只包含这 5 个变量的矩阵散点图，我们键入：

```
pd.tools.plotting.scatter_matrix(data.loc[:, "V2":"V6"], diagonal="kde")
plt.tight_layout()
plt.show()

```

![png](img/a_little_book_of_python_for_multivariate_analysis_17_0.png)

在这个矩阵散点图中，对角线单元格显示了每个变量的直方图，在本例中是前五种化学品的浓度（变量 V2、V3、V4、V5、V6）。

每个非对角单元格都是五种化学品中的两种之间的散点图，例如，第一行中的第二个单元格是 V2（y 轴）与 V3（x 轴）的散点图。### 通过其组标记的散点图

如果在矩阵散点图中看到两个变量的有趣散点图，你可能想详细绘制该散点图，数据点标记为它们的组别（在本例中为它们的培育品种）。

例如，在上述矩阵散点图中，第三列第四行下的单元格是 V5（x 轴）与 V4（y 轴）的散点图。如果你看这个散点图，似乎 V5 和 V4 之间可能存在正相关关系。

因此，我们可能决定更仔细地研究 V5 和 V4 之间的关系，通过绘制这两个变量的散点图，并将数据点标记为它们的组别（在本例中为它们的培育品种）。要绘制两个变量的散点图，我们可以使用`seaborn`包中的`lmplot`函数。变量 V4 和 V5 存储在变量`data`的列 V4 和 V5 中。`lmplot()`函数中的前两个参数是要在 x-y 中绘制的列，第三个参数指定数据，`hue`参数是用于数据点标签的列名，即它们所属的类，最后，当我们不想绘制与 x-y 变量相关的回归模型时，`fit_reg`参数设置为`False`。因此，要绘制散点图，我们键入：

```
sns.lmplot("V4", "V5", data, hue="V1", fit_reg=False);

```

![png](img/a_little_book_of_python_for_multivariate_analysis_20_0.png)

从 V4 与 V5 的散点图中可以看出，培育品种 2 的葡萄酒似乎具有较低的 V4 值，而与培育品种 1 的葡萄酒相比。### 个人资料图

另一种有用的绘图类型是*个人资料图*，它通过绘制每个样本的每个变量的值来显示每个变量的变化。

这可以通过使用`pandas`的绘图功能来实现，它建立在`matplotlib`之上，通过运行以下命令：

```
ax = data[["V2","V3","V4","V5","V6"]].plot()
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5));

```

![png](img/a_little_book_of_python_for_multivariate_analysis_23_0.png)

从概要图中可以清楚地看出，V6 的均值和标准差比其他变量要高得多。## 计算多元数据的摘要统计

您可能想要做的另一件事是计算多元数据集中每个变量的摘要统计，例如均值和标准差。

这很容易做到，使用`numpy`中的`mean()`和`std()`函数，并将它们应用于数据框架，使用其成员函数`apply`。

Pandas 允许直接调用简单操作作为方法，例如我们可以通过调用`df.mean()`计算数据框架`df`的均值。

另一种选择是使用`pandas.DataFrame`类的`apply`方法，它沿着数据框架的输入轴应用传递的参数函数。这种方法非常强大，因为它允许传递任何我们想要应用于数据的函数。

例如，假设我们想计算葡萄酒样本中每种 13 种化学物质浓度的均值和标准差。这些存储在变量`data`的列 V2-V14 中，之前为方便起见已经分配给`X`。因此我们输入：

```
X.apply(np.mean)

```

```
V2      13.000618
V3       2.336348
V4       2.366517
V5      19.494944
V6      99.741573
V7       2.295112
V8       2.029270
V9       0.361854
V10      1.590899
V11      5.058090
V12      0.957449
V13      2.611685
V14    746.893258
dtype: float64

```

这告诉我们，变量 V2 的均值为 13.000618，V3 的均值为 2.336348，依此类推。

类似地，要获取 13 种化学物质浓度的标准差，我们输入：

```
X.apply(np.std)

```

```
V2       0.809543
V3       1.114004
V4       0.273572
V5       3.330170
V6      14.242308
V7       0.624091
V8       0.996049
V9       0.124103
V10      0.570749
V11      2.311765
V12      0.227929
V13      0.707993
V14    314.021657
dtype: float64

```

我们可以看到，为了比较变量，标准化是有意义的，因为变量的标准差非常不同 - V14 的标准差为 314.021657，而 V9 的标准差仅为 0.124103。因此，为了比较变量，我们需要标准化每个变量，使其具有样本方差为 1 和样本均值为 0。我们将在下面解释如何标准化变量。

### 每组的均值和方差

计算每个组样本的均值和标准差通常很有趣，例如，对于每个品种的葡萄酒样本。品种存储在变量`data`的列 V1 中，之前为方便起见已经分配给`y`。

要提取出仅仅是品种 2 的数据，我们可以输入：

```
class2data = data[y=="2"]

```

然后，我们可以计算 13 种化学物质浓度的均值和标准差，仅针对品种 2 样本：

```
class2data.loc[:, "V2":].apply(np.mean)

```

```
V2      12.278732
V3       1.932676
V4       2.244789
V5      20.238028
V6      94.549296
V7       2.258873
V8       2.080845
V9       0.363662
V10      1.630282
V11      3.086620
V12      1.056282
V13      2.785352
V14    519.507042
dtype: float64

```

```
class2data.loc[:, "V2":].apply(np.std)

```

```
V2       0.534162
V3       1.008391
V4       0.313238
V5       3.326097
V6      16.635097
V7       0.541507
V8       0.700713
V9       0.123085
V10      0.597813
V11      0.918393
V12      0.201503
V13      0.493064
V14    156.100173
dtype: float64

```

您可以类似地计算 13 种化学物质浓度的均值和标准差，仅针对品种 1 样本，或仅针对品种 3 样本。

然而，为了方便起见，您可能希望使用下面的函数`printMeanAndSdByGroup()`，它会打印出数据集中每个组的变量的均值和标准差：

```
def printMeanAndSdByGroup(variables, groupvariable):
    data_groupby = variables.groupby(groupvariable)
    print("## Means:")
    display(data_groupby.apply(np.mean))
    print("\n## Standard deviations:")
    display(data_groupby.apply(np.std))
    print("\n## Sample sizes:")
    display(pd.DataFrame(data_groupby.apply(len)))

```

函数的参数是您要计算均值和标准差的变量（`X`），以及包含每个样本组的变量（`y`）。例如，要计算 13 种化学浓度中每种酒品种的平均值和标准差，我们可以输入：

```
printMeanAndSdByGroup(X, y)

```

```
## Means:

```

|  | V2 | V3 | V4 | V5 | V6 | V7 | V8 | V9 | V10 | V11 | V12 | V13 | V14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V1 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 13.744746 | 2.010678 | 2.455593 | 17.037288 | 106.338983 | 2.840169 | 2.982373 | 0.290000 | 1.899322 | 5.528305 | 1.062034 | 3.157797 | 1115.711864 |
| 2 | 12.278732 | 1.932676 | 2.244789 | 20.238028 | 94.549296 | 2.258873 | 2.080845 | 0.363662 | 1.630282 | 3.086620 | 1.056282 | 2.785352 | 519.507042 |
| 3 | 13.153750 | 3.333750 | 2.437083 | 21.416667 | 99.312500 | 1.678750 | 0.781458 | 0.447500 | 1.153542 | 7.396250 | 0.682708 | 1.683542 | 629.895833 |

```
## Standard deviations:

```

|  | V2 | V3 | V4 | V5 | V6 | V7 | V8 | V9 | V10 | V11 | V12 | V13 | V14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V1 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.458192 | 0.682689 | 0.225233 | 2.524651 | 10.409595 | 0.336077 | 0.394111 | 0.069453 | 0.408602 | 1.228032 | 0.115491 | 0.354038 | 219.635449 |
| 2 | 0.534162 | 1.008391 | 0.313238 | 3.326097 | 16.635097 | 0.541507 | 0.700713 | 0.123085 | 0.597813 | 0.918393 | 0.201503 | 0.493064 | 156.100173 |
| 3 | 0.524689 | 1.076514 | 0.182756 | 2.234515 | 10.776433 | 0.353233 | 0.290431 | 0.122840 | 0.404555 | 2.286743 | 0.113243 | 0.269262 | 113.891805 |

```
## Sample sizes:

```

|  | 0 |
| --- | --- |
| V1 |  |
| --- | --- |
| 1 | 59 |
| 2 | 71 |
| 3 | 48 |

函数`printMeanAndSdByGroup()`还会输出每个组中的样本数量。在这个例子中，我们看到品种 1 有 59 个样本，品种 2 有 71 个样本，品种 3 有 48 个样本。### 变量的组间方差和组内方差

如果我们想要计算特定变量的组内方差（例如，某种化学物质的浓度），我们可以使用下面的函数`calcWithinGroupsVariance()`：

```
def calcWithinGroupsVariance(variable, groupvariable):
    # find out how many values the group variable can take
    levels = sorted(set(groupvariable))
    numlevels = len(levels)
    # get the mean and standard deviation for each group:
    numtotal = 0
    denomtotal = 0
    for leveli in levels:
        levelidata = variable[groupvariable==leveli]
        levelilength = len(levelidata)
        # get the standard deviation for group i:
        sdi = np.std(levelidata)
        numi = (levelilength)*sdi**2
        denomi = levelilength
        numtotal = numtotal + numi
        denomtotal = denomtotal + denomi
    # calculate the within-groups variance
    Vw = numtotal / (denomtotal - numlevels)
    return Vw

```

函数`calcWithinGroupsVariance()`的`variable`参数是我们希望为`groupvariable`中给定的组计算其组内方差的输入变量。

例如，要计算变量 V2（第一种化学物质的浓度）的组内方差，我们可以输入：

```
calcWithinGroupsVariance(X.V2, y)

```

```
0.2620524691539065

```

因此，变量 V2 的组内方差为 0.2620525。

我们可以使用下面的函数`calcBetweenGroupsVariance()`来计算特定变量（例如 V2）的组间方差：

```
def calcBetweenGroupsVariance(variable, groupvariable):
    # find out how many values the group variable can take
    levels = sorted(set((groupvariable)))
    numlevels = len(levels)
    # calculate the overall grand mean:
    grandmean = np.mean(variable)
    # get the mean and standard deviation for each group:
    numtotal = 0
    denomtotal = 0
    for leveli in levels:
        levelidata = variable[groupvariable==leveli]
        levelilength = len(levelidata)
        # get the mean and standard deviation for group i:
        meani = np.mean(levelidata)
        sdi = np.std(levelidata)
        numi = levelilength * ((meani - grandmean)**2)
        denomi = levelilength
        numtotal = numtotal + numi
        denomtotal = denomtotal + denomi
    # calculate the between-groups variance
    Vb = numtotal / (numlevels - 1)
    return(Vb)

```

与函数`calcWithinGroupsVariance()`的参数类似，函数`calcBetweenGroupsVariance()`的`variable`参数是我们希望计算其组间方差的输入变量，给定在`groupvariable`中的组。

因此，例如，要计算变量 V2（第一种化学品的浓度）的组间方差，我们输入：

```
calcBetweenGroupsVariance(X.V2, y)

```

```
35.397424960269106

```

因此，V2 的组间方差为 35.397425。

我们可以通过一个变量的组间方差除以其组内方差来计算其*分离度*。因此，V2 实现的分离度计算如下：

```
# 35.397424960269106 / 0.2620524691539065
calcBetweenGroupsVariance(X.V2, y) / calcWithinGroupsVariance(X.V2, y)

```

```
135.07762424279917

```

如果要计算多元数据集中所有变量实现的分离度，可以使用下面的函数`calcSeparations()`：

```
def calcSeparations(variables, groupvariable):
    # calculate the separation for each variable
    for variablename in variables:
        variablei = variables[variablename]
        Vw = calcWithinGroupsVariance(variablei, groupvariable)
        Vb = calcBetweenGroupsVariance(variablei, groupvariable)
        sep = Vb/Vw
        print("variable", variablename, "Vw=", Vw, "Vb=", Vb, "separation=", sep)

```

例如，要计算 13 种化学品浓度的分离度，我们输入：

```
calcSeparations(X, y)

```

```
variable V2 Vw= 0.262052469154 Vb= 35.3974249603 separation= 135.077624243
variable V3 Vw= 0.887546796747 Vb= 32.7890184869 separation= 36.9434249632
variable V4 Vw= 0.0660721013425 Vb= 0.879611357249 separation= 13.3129012
variable V5 Vw= 8.00681118121 Vb= 286.416746363 separation= 35.7716374073
variable V6 Vw= 180.657773164 Vb= 2245.50102789 separation= 12.4295843381
variable V7 Vw= 0.191270475224 Vb= 17.9283572943 separation= 93.7330096204
variable V8 Vw= 0.274707514337 Vb= 64.2611950236 separation= 233.925872682
variable V9 Vw= 0.0119117022133 Vb= 0.328470157462 separation= 27.575417147
variable V10 Vw= 0.246172943796 Vb= 7.45199550778 separation= 30.2713831702
variable V11 Vw= 2.28492308133 Vb= 275.708000822 separation= 120.664018441
variable V12 Vw= 0.0244876469432 Vb= 2.48100991494 separation= 101.31679539
variable V13 Vw= 0.160778729561 Vb= 30.5435083544 separation= 189.972320579
variable V14 Vw= 29707.6818705 Vb= 6176832.32228 separation= 207.920373902

```

因此，使得组（葡萄酒品种）之间分离度最大的单个变量是 V8（分离度 233.9）。正如我们将在下文讨论的，线性判别分析（LDA）的目的是找到个体变量的线性组合，以实现组（这里是品种）之间最大的分离度。这有望比任何单个变量实现的最佳分离度（这里是 V8 的 233.9）更好。### 两个变量的组间协方差和组内协方差

如果您有一个描述来自不同组的采样单位的多元数据集，例如来自不同品种的葡萄酒样本，通常有兴趣计算一对变量的组内协方差和组间方差。

这可以通过以下函数完成：

```
def calcWithinGroupsCovariance(variable1, variable2, groupvariable):
    levels = sorted(set(groupvariable))
    numlevels = len(levels)
    Covw = 0.0
    # get the covariance of variable 1 and variable 2 for each group:
    for leveli in levels:
        levelidata1 = variable1[groupvariable==leveli]
        levelidata2 = variable2[groupvariable==leveli]
        mean1 = np.mean(levelidata1)
        mean2 = np.mean(levelidata2)
        levelilength = len(levelidata1)
        # get the covariance for this group:
        term1 = 0.0
        for levelidata1j, levelidata2j in zip(levelidata1, levelidata2):
            term1 += (levelidata1j - mean1)*(levelidata2j - mean2)
        Cov_groupi = term1 # covariance for this group
        Covw += Cov_groupi
    totallength = len(variable1)
    Covw /= totallength - numlevels
    return Covw

```

例如，要计算变量 V8 和 V11 的组内协方差，我们输入：

```
calcWithinGroupsCovariance(X.V8, X.V11, y)

```

```
0.28667830215140183

```

```
def calcBetweenGroupsCovariance(variable1, variable2, groupvariable):
    # find out how many values the group variable can take
    levels = sorted(set(groupvariable))
    numlevels = len(levels)
    # calculate the grand means
    variable1mean = np.mean(variable1)
    variable2mean = np.mean(variable2)
    # calculate the between-groups covariance
    Covb = 0.0
    for leveli in levels:
        levelidata1 = variable1[groupvariable==leveli]
        levelidata2 = variable2[groupvariable==leveli]
        mean1 = np.mean(levelidata1)
        mean2 = np.mean(levelidata2)
        levelilength = len(levelidata1)
        term1 = (mean1 - variable1mean) * (mean2 - variable2mean) * levelilength
        Covb += term1
    Covb /= numlevels - 1
    return Covb

```

例如，要计算变量 V8 和 V11 的组间协方差，我们输入：

```
calcBetweenGroupsCovariance(X.V8, X.V11, y)

```

```
-60.4107748359163

```

因此，对于 V8 和 V11，组间协方差为-60.41，组内协方差为 0.29。由于组内协方差为正值（0.29），这意味着 V8 和 V11 在组内呈正相关：对于同一组的个体，V8 值高的个体往往具有较高的 V11 值，反之亦然。由于组间协方差为负值（-60.41），V8 和 V11 在组间呈负相关：具有较高平均 V8 值的组往往具有较低平均 V11 值，反之亦然。### 计算多元数据的相关性

通常有兴趣研究多元数据集中的变量是否存在显著相关性。

要计算一对变量的线性（皮尔逊）相关系数，可以使用`scipy.stats`包中的`pearsonr()`函数。例如，要计算前两种化学品浓度 V2 和 V3 的相关系数，我们输入：

```
corr = stats.pearsonr(X.V2, X.V3)
print("p-value:\t", corr[1])
print("cor:\t\t", corr[0])

```

```
p-value:     0.210081985971
cor:         0.0943969409104

```

这告诉我们相关系数约为 0.094，这是一个非常弱的相关性。此外，用于检验相关系数是否显著不同于零的统计检验的*p 值*为 0.21。这远大于 0.05（我们可以将其作为统计显著性的截止值），因此几乎没有证据表���相关性不为零。

如果你有很多变量，可以使用`pandas.DataFrame`方法`corr()`来计算一个相关矩阵，显示每对变量之间的相关系数。

```
corrmat = X.corr()
corrmat

```

|  | V2 | V3 | V4 | V5 | V6 | V7 | V8 | V9 | V10 | V11 | V12 | V13 | V14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V2 | 1.000000 | 0.094397 | 0.211545 | -0.310235 | 0.270798 | 0.289101 | 0.236815 | -0.155929 | 0.136698 | 0.546364 | -0.071747 | 0.072343 | 0.643720 |
| V3 | 0.094397 | 1.000000 | 0.164045 | 0.288500 | -0.054575 | -0.335167 | -0.411007 | 0.292977 | -0.220746 | 0.248985 | -0.561296 | -0.368710 | -0.192011 |
| V4 | 0.211545 | 0.164045 | 1.000000 | 0.443367 | 0.286587 | 0.128980 | 0.115077 | 0.186230 | 0.009652 | 0.258887 | -0.074667 | 0.003911 | 0.223626 |
| V5 | -0.310235 | 0.288500 | 0.443367 | 1.000000 | -0.083333 | -0.321113 | -0.351370 | 0.361922 | -0.197327 | 0.018732 | -0.273955 | -0.276769 | -0.440597 |
| V6 | 0.270798 | -0.054575 | 0.286587 | -0.083333 | 1.000000 | 0.214401 | 0.195784 | -0.256294 | 0.236441 | 0.199950 | 0.055398 | 0.066004 | 0.393351 |
| V7 | 0.289101 | -0.335167 | 0.128980 | -0.321113 | 0.214401 | 1.000000 | 0.864564 | -0.449935 | 0.612413 | -0.055136 | 0.433681 | 0.699949 | 0.498115 |
| V8 | 0.236815 | -0.411007 | 0.115077 | -0.351370 | 0.195784 | 0.864564 | 1.000000 | -0.537900 | 0.652692 | -0.172379 | 0.543479 | 0.787194 | 0.494193 |
| V9 | -0.155929 | 0.292977 | 0.186230 | 0.361922 | -0.256294 | -0.449935 | -0.537900 | 1.000000 | -0.365845 | 0.139057 | -0.262640 | -0.503270 | -0.311385 |
| V10 | 0.136698 | -0.220746 | 0.009652 | -0.197327 | 0.236441 | 0.612413 | 0.652692 | -0.365845 | 1.000000 | -0.025250 | 0.295544 | 0.519067 | 0.330417 |
| V11 | 0.546364 | 0.248985 | 0.258887 | 0.018732 | 0.199950 | -0.055136 | -0.172379 | 0.139057 | -0.025250 | 1.000000 | -0.521813 | -0.428815 | 0.316100 |
| V12 | -0.071747 | -0.561296 | -0.074667 | -0.273955 | 0.055398 | 0.433681 | 0.543479 | -0.262640 | 0.295544 | -0.521813 | 1.000000 | 0.565468 | 0.236183 |
| V13 | 0.072343 | -0.368710 | 0.003911 | -0.276769 | 0.066004 | 0.699949 | 0.787194 | -0.503270 | 0.519067 | -0.428815 | 0.565468 | 1.000000 | 0.312761 |
| V14 | 0.643720 | -0.192011 | 0.223626 | -0.440597 | 0.393351 | 0.498115 | 0.494193 | -0.311385 | 0.330417 | 0.316100 | 0.236183 | 0.312761 | 1.000000 |

通过*热图*形式的相关矩阵图，可以更好地展示相关矩阵的相关性。

```
sns.heatmap(corrmat, vmax=1., square=False).xaxis.tick_top()

```

![png](img/a_little_book_of_python_for_multivariate_analysis_68_0.png)

或者另一种很好的可视化方式是通过 Hinton 图。 方框的颜色确定相关性的符号，本例中红色表示正相关，蓝色表示负相关； 而方框的大小确定其大小，方框越大，其数量越大。

```
# adapted from http://matplotlib.org/examples/specialty_plots/hinton_demo.html
def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2**np.ceil(np.log(np.abs(matrix).max())/np.log(2))

    ax.patch.set_facecolor('lightgray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'red' if w > 0 else 'blue'
        size = np.sqrt(np.abs(w))
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    nticks = matrix.shape[0]
    ax.xaxis.tick_top()
    ax.set_xticks(range(nticks))
    ax.set_xticklabels(list(matrix.columns), rotation=90)
    ax.set_yticks(range(nticks))
    ax.set_yticklabels(matrix.columns)
    ax.grid(False)

    ax.autoscale_view()
    ax.invert_yaxis()

hinton(corrmat)

```

![png](img/a_little_book_of_python_for_multivariate_analysis_70_0.png)

尽管相关矩阵和图表对于快速查找识别最强相关性非常有用，但仍需要劳动工作来查找前 N 个最强相关性。 为此，您可以使用下面的函数 `mosthighlycorrelated()`。

函数 `mosthighlycorrelated()` 将按照相关系数的顺序打印出数据集中每对变量的线性相关系数。 这让您非常容易地看到哪对变量是最高度相关的。

```
def mosthighlycorrelated(mydataframe, numtoreport):
    # find the correlations
    cormatrix = mydataframe.corr()
    # set the correlations on the diagonal or lower triangle to zero,
    # so they will not be reported as the highest ones:
    cormatrix *= np.tri(*cormatrix.values.shape, k=-1).T
    # find the top n correlations
    cormatrix = cormatrix.stack()
    cormatrix = cormatrix.reindex(cormatrix.abs().sort_values(ascending=False).index).reset_index()
    # assign human-friendly names
    cormatrix.columns = ["FirstVariable", "SecondVariable", "Correlation"]
    return cormatrix.head(numtoreport)

```

函数的参数是要计算相关性的变量，以及要打印出的前 N 个相关系数的数量（例如，您可以告诉它打印出最大的 10 个相关系数或最大的 20 个相关系数）。

例如，要计算葡萄酒样品中 13 种化学物质浓度之间的相关系数，并打印出前 10 个成对相关系数，您可以键入：

```
mosthighlycorrelated(X, 10)

```

|  | 第一个变量 | 第二个变量 | 相关系数 |
| --- | --- | --- | --- |
| 0 | V7 | V8 | 0.864564 |
| 1 | V8 | V13 | 0.787194 |
| 2 | V7 | V13 | 0.699949 |
| 3 | V8 | V10 | 0.652692 |
| 4 | V2 | V14 | 0.643720 |
| 5 | V7 | V10 | 0.612413 |
| 6 | V12 | V13 | 0.565468 |
| 7 | V3 | V12 | -0.561296 |
| 8 | V2 | V11 | 0.546364 |
| 9 | V8 | V12 | 0.543479 |

这告诉我们，具有最高线性相关系数的变量对是 V7 和 V8（相关性约为 0.86）。 ### 标准化变量

如果您想比较具有不同单位的不同变量，方差非常不同，那么首先标准化变量是一个好主意。

例如，我们发现上面葡萄酒样品中的 13 种化学物质浓度显示出很大范围的标准偏差，从 V9 的 0.124103（方差 0.015402）到 V14 的 314.021657（方差 98609.60）。 这是方差中约 6,402,389 倍的范围。

因此，不建议使用未标准化的化学浓度作为葡萄酒样品主成分分析（PCA，请参见下文）的输入，因为如果这样做，第一个主成分将由显示最大方差的变量主导，例如 V14。

因此，首先将变量标准化为方差为 1，均值为 0 可能是一个更好的主意，然后在标准化数据上进行主成分分析。这样可以使我们找到提供原始数据变化最佳低维表示的主成分，而不会被那些在原始数据中显示最大方差的变量过度偏置。

您可以使用`sklearn.preprocessing`包中的`scale()`函数对变量进行标准化。

例如，要对葡萄酒样本中的 13 种化学物质的浓度进行标准化，我们键入：

```
standardisedX = scale(X)
standardisedX = pd.DataFrame(standardisedX, index=X.index, columns=X.columns)

```

```
standardisedX.apply(np.mean)

```

```
V2    -8.619821e-16
V3    -8.357859e-17
V4    -8.657245e-16
V5    -1.160121e-16
V6    -1.995907e-17
V7    -2.972030e-16
V8    -4.016762e-16
V9     4.079134e-16
V10   -1.699639e-16
V11   -1.247442e-18
V12    3.717376e-16
V13    2.919013e-16
V14   -7.484650e-18
dtype: float64

```

```
standardisedX.apply(np.std)

```

```
V2     1
V3     1
V4     1
V5     1
V6     1
V7     1
V8     1
V9     1
V10    1
V11    1
V12    1
V13    1
V14    1
dtype: float64

```  ## 主成分分析

主成分分析的目的是找到多变量数据集变化的最佳低维表示。例如，在葡萄酒数据集的情况下，我们有 13 种描述来自三种不同品种葡萄酒样本的化学浓度。我们可以进行主成分分析，以调查我们是否可以使用更少的新变量（主成分）捕获样本之间的大部分变化，其中每个新变量都是所有或部分 13 种化学浓度的线性组合。

要对多变量数据集进行主成分分析（PCA），通常第一步是使用`scale()`函数对研究中的变量进行标准化（请参见上文）。如果输入变量的方差非常不同，则有必要进行这样的标准化，这在这种情况下是正确的，因为 13 种化学品的浓度具有非常不同的方差（请参见上文）。

一旦您标准化了变量，就可以使用`sklearn.decomposition`包中的`PCA`类及其`fit`方法进行主成分分析，该方法将数据`X`与模型拟合。默认的`solver`是奇异值分解（“svd”）。有关更多信息，您可以在 Python 控制台中键入`help(PCA)`。

例如，要对葡萄酒样本中的 13 种化学物质的浓度进行标准化，并对标准化的浓度进行主成分分析，我们键入：

```
pca = PCA().fit(standardisedX)

```

您可以使用下面的`pca_summary()`函数来获取主成分分析结果的摘要，该函数模拟了 R 的`summary`函数对 PCA 模型的输出：

```
def pca_summary(pca, standardised_data, out=True):
    names = ["PC"+str(i) for i in range(1, len(pca.explained_variance_ratio_)+1)]
    a = list(np.std(pca.transform(standardised_data), axis=0))
    b = list(pca.explained_variance_ratio_)
    c = [np.sum(pca.explained_variance_ratio_[:i]) for i in range(1, len(pca.explained_variance_ratio_)+1)]
    columns = pd.MultiIndex.from_tuples([("sdev", "Standard deviation"), ("varprop", "Proportion of Variance"), ("cumprop", "Cumulative Proportion")])
    summary = pd.DataFrame(zip(a, b, c), index=names, columns=columns)
    if out:
        print("Importance of components:")
        display(summary)
    return summary

```

`print_pca_summary`函数的参数是：

+   `pca`：一个 PCA 对象

+   `standardised_data`：标准化后的数据

+   `out (True)`: 输出到标准输出

```
summary = pca_summary(pca, standardisedX)

```

```
Importance of components:

```

|  | 标准差 | 方差比 | 累积比 |
| --- | --- | --- | --- |
|  | 标准差 | 方差比例 | 累积比例 |
| --- | --- | --- | --- |
| PC1 | 2.169297 | 0.361988 | 0.361988 |
| PC2 | 1.580182 | 0.192075 | 0.554063 |
| PC3 | 1.202527 | 0.111236 | 0.665300 |
| PC4 | 0.958631 | 0.070690 | 0.735990 |
| PC5 | 0.923704 | 0.065633 | 0.801623 |
| PC6 | 0.801035 | 0.049358 | 0.850981 |
| PC7 | 0.742313 | 0.042387 | 0.893368 |
| PC8 | 0.590337 | 0.026807 | 0.920175 |
| PC9 | 0.537476 | 0.022222 | 0.942397 |
| PC10 | 0.500902 | 0.019300 | 0.961697 |
| PC11 | 0.475172 | 0.017368 | 0.979066 |
| PC12 | 0.410817 | 0.012982 | 0.992048 |
| PC13 | 0.321524 | 0.007952 | 1.000000 |

这给出了每个组分的标准偏差，以及每个组分解释的方差比例。组分的标准偏差存储在由`pca_summary`函数生成的输出变量的名为`sdev`的命名行中，并存储在`summary`变量中：

```
summary.sdev

```

|  | 标准偏差 |
| --- | --- |
| PC1 | 2.169297 |
| PC2 | 1.580182 |
| PC3 | 1.202527 |
| PC4 | 0.958631 |
| PC5 | 0.923704 |
| PC6 | 0.801035 |
| PC7 | 0.742313 |
| PC8 | 0.590337 |
| PC9 | 0.537476 |
| PC10 | 0.500902 |
| PC11 | 0.475172 |
| PC12 | 0.410817 |
| PC13 | 0.321524 |

总方差由各组分的方差之和得出：

```
np.sum(summary.sdev**2)

```

```
Standard deviation    13
dtype: float64

```

在这种情况下，我们看到总方差为 13，等于标准化变量的数量（13 个变量）。这是因为对于标准化数据，每个标准化变量的方差为 1。总方差等于各个变量的方差之和，而每个标准化变量的方差为 1，因此总方差应等于变量的数量（此处为 13）。

### 决定保留多少主成分

为了决定应保留多少主成分，通常可以通过制作一张屏幕图来总结主成分分析的结果，我们可以使用下面的`screeplot()`函数来实现：

```
def screeplot(pca, standardised_values):
    y = np.std(pca.transform(standardised_values), axis=0)**2
    x = np.arange(len(y)) + 1
    plt.plot(x, y, "o-")
    plt.xticks(x, ["Comp."+str(i) for i in x], rotation=60)
    plt.ylabel("Variance")
    plt.show()

screeplot(pca, standardisedX)

```

![png](img/a_little_book_of_python_for_multivariate_analysis_92_0.png)

屏幕图中最明显的斜率变化发生在第 4 个组分，这是屏幕图的“拐点”。因此，基于屏幕图的基础可以主张保留前三个组分。

另一种决定保留多少成分的方法是使用*Kaiser 准则*：我们只应保留方差大于 1 的主成分（在对标准化数据进行主成分分析时）。我们可以通过找到每个主成分的方差来检查这一点：

```
summary.sdev**2

```

|  | 标准偏差 |
| --- | --- |
| PC1 | 4.705850 |
| PC2 | 2.496974 |
| PC3 | 1.446072 |
| PC4 | 0.918974 |
| PC5 | 0.853228 |
| PC6 | 0.641657 |
| PC7 | 0.551028 |
| PC8 | 0.348497 |
| PC9 | 0.288880 |
| PC10 | 0.250902 |
| PC11 | 0.225789 |
| PC12 | 0.168770 |
| PC13 | 0.103378 |

我们看到主成分 1、2 和 3 的方差都大于 1（分别为 4.71、2.50 和 1.45）。因此，根据 Kaiser 准则，我们将保留前三个主成分。

决定保留多少主成分的第三种方法是决定保留至少解释总方差的一定最小比例的主成分数量。例如，如果重要的是解释至少 80%的方差，我们将保留前五个主成分，因为我们可以从累积比例（`summary.cumprop`）中看到，前五个主成分解释了 80.2%的方差（而前四个主成分仅解释了 73.6%，因此不足够）。### 主成分的载荷

主成分的载荷存储在由`PCA().fit()`返回的变量的命名元素`components_`中。这包含一个矩阵，其中包含每个主成分的载荷，矩阵的第一列包含第一个主成分的载荷，第二列包含第二个主成分的载荷，依此类推。

因此，在我们对葡萄酒样本中的 13 种化学浓度进行分析时，要获得第一个主成分的载荷，我们输入：

```
pca.components_[0]

```

```
array([-0.1443294 ,  0.24518758,  0.00205106,  0.23932041, -0.14199204,
       -0.39466085, -0.4229343 ,  0.2985331 , -0.31342949,  0.0886167 ,
       -0.29671456, -0.37616741, -0.28675223])

```

这意味着第一个主成分是变量的线性组合：

```
-0.144*Z2 + 0.245*Z3 + 0.002*Z4 + 0.239*Z5 - 0.142*Z6 - 0.395*Z7 - 0.423*Z8 + 0.299*Z9 -0.313*Z10 + 0.089*Z11 - 0.297*Z12 - 0.376*Z13 - 0.287*Z14

```

其中 Z2、Z3、Z4，...，Z14 是变量 V2、V3、V4，...，V14 的标准化版本（每个版本的均值为 0，方差为 1）。

请注意，载荷的平方和为 1，因为这是计算载荷时使用的约束条件：

```
np.sum(pca.components_[0]**2)

```

```
1.0000000000000004

```

要计算第一个主成分的值，我们可以定义自己的函数，根据载荷和输入变量的值来计算主成分：

```
def calcpc(variables, loadings):
    # find the number of samples in the data set and the number of variables
    numsamples, numvariables = variables.shape
    # make a vector to store the component
    pc = np.zeros(numsamples)
    # calculate the value of the component for each sample
    for i in range(numsamples):
        valuei = 0
        for j in range(numvariables):
            valueij = variables.iloc[i, j]
            loadingj = loadings[j]
            valuei = valuei + (valueij * loadingj)
        pc[i] = valuei
    return pc

```

然后，我们可以使用该函数计算我们葡萄酒数据中每个样本的第一个主成分的值：

```
calcpc(standardisedX, pca.components_[0])

```

```
array([-3.31675081, -2.20946492, -2.51674015, -3.75706561, -1.00890849,
       -3.05025392, -2.44908967, -2.05943687, -2.5108743 , -2.75362819,
       -3.47973668, -1.7547529 , -2.11346234, -3.45815682, -4.31278391,
       -2.3051882 , -2.17195527, -1.89897118, -3.54198508, -2.0845222 ,
       -3.12440254, -1.08657007, -2.53522408, -1.64498834, -1.76157587,
       -0.9900791 , -1.77527763, -1.23542396, -2.18840633, -2.25610898,
       -2.50022003, -2.67741105, -1.62857912, -1.90269086, -1.41038853,
       -1.90382623, -1.38486223, -1.12220741, -1.5021945 , -2.52980109,
       -2.58809543, -0.66848199, -3.07080699, -0.46220914, -2.10135193,
       -1.13616618, -2.72660096, -2.82133927, -2.00985085, -2.7074913 ,
       -3.21491747, -2.85895983, -3.50560436, -2.22479138, -2.14698782,
       -2.46932948, -2.74151791, -2.17374092, -3.13938015,  0.92858197,
        1.54248014,  1.83624976, -0.03060683, -2.05026161,  0.60968083,
       -0.90022784, -2.24850719, -0.18338403,  0.81280503, -1.9756205 ,
        1.57221622, -1.65768181,  0.72537239, -2.56222717, -1.83256757,
        0.8679929 , -0.3700144 ,  1.45737704, -1.26293085, -0.37615037,
       -0.7620639 , -1.03457797,  0.49487676,  2.53897708, -0.83532015,
       -0.78790461,  0.80683216,  0.55804262,  1.11511104,  0.55572283,
        1.34928528,  1.56448261,  1.93255561, -0.74666594, -0.95745536,
       -2.54386518,  0.54395259, -1.03104975, -2.25190942, -1.41021602,
       -0.79771979,  0.54953173,  0.16117374,  0.65979494, -0.39235441,
        1.77249908,  0.36626736,  1.62067257, -0.08253578, -1.57827507,
       -1.42056925,  0.27870275,  1.30314497,  0.45707187,  0.49418585,
       -0.48207441,  0.25288888,  0.10722764,  2.4330126 ,  0.55108954,
       -0.73962193, -1.33632173,  1.177087  ,  0.46233501, -0.97847408,
        0.09680973, -0.03848715,  1.5971585 ,  0.47956492,  1.79283347,
        1.32710166,  2.38450083,  2.9369401 ,  2.14681113,  2.36986949,
        3.06384157,  3.91575378,  3.93646339,  3.09427612,  2.37447163,
        2.77881295,  2.28656128,  2.98563349,  2.3751947 ,  2.20986553,
        2.625621  ,  4.28063878,  3.58264137,  2.80706372,  2.89965933,
        2.32073698,  2.54983095,  1.81254128,  2.76014464,  2.7371505 ,
        3.60486887,  2.889826  ,  3.39215608,  1.0481819 ,  1.60991228,
        3.14313097,  2.2401569 ,  2.84767378,  2.59749706,  2.94929937,
        3.53003227,  2.40611054,  2.92908473,  2.18141278,  2.38092779,
        3.21161722,  3.67791872,  2.4655558 ,  3.37052415,  2.60195585,
        2.67783946,  2.38701709,  3.20875816])

```

实际上，第一个主成分的值是用以下方式计算的，因此我们可以将这些值与我们计算的值进行比较，它们应该一致：

```
pca.transform(standardisedX)[:, 0]

```

```
array([-3.31675081, -2.20946492, -2.51674015, -3.75706561, -1.00890849,
       -3.05025392, -2.44908967, -2.05943687, -2.5108743 , -2.75362819,
       -3.47973668, -1.7547529 , -2.11346234, -3.45815682, -4.31278391,
       -2.3051882 , -2.17195527, -1.89897118, -3.54198508, -2.0845222 ,
       -3.12440254, -1.08657007, -2.53522408, -1.64498834, -1.76157587,
       -0.9900791 , -1.77527763, -1.23542396, -2.18840633, -2.25610898,
       -2.50022003, -2.67741105, -1.62857912, -1.90269086, -1.41038853,
       -1.90382623, -1.38486223, -1.12220741, -1.5021945 , -2.52980109,
       -2.58809543, -0.66848199, -3.07080699, -0.46220914, -2.10135193,
       -1.13616618, -2.72660096, -2.82133927, -2.00985085, -2.7074913 ,
       -3.21491747, -2.85895983, -3.50560436, -2.22479138, -2.14698782,
       -2.46932948, -2.74151791, -2.17374092, -3.13938015,  0.92858197,
        1.54248014,  1.83624976, -0.03060683, -2.05026161,  0.60968083,
       -0.90022784, -2.24850719, -0.18338403,  0.81280503, -1.9756205 ,
        1.57221622, -1.65768181,  0.72537239, -2.56222717, -1.83256757,
        0.8679929 , -0.3700144 ,  1.45737704, -1.26293085, -0.37615037,
       -0.7620639 , -1.03457797,  0.49487676,  2.53897708, -0.83532015,
       -0.78790461,  0.80683216,  0.55804262,  1.11511104,  0.55572283,
        1.34928528,  1.56448261,  1.93255561, -0.74666594, -0.95745536,
       -2.54386518,  0.54395259, -1.03104975, -2.25190942, -1.41021602,
       -0.79771979,  0.54953173,  0.16117374,  0.65979494, -0.39235441,
        1.77249908,  0.36626736,  1.62067257, -0.08253578, -1.57827507,
       -1.42056925,  0.27870275,  1.30314497,  0.45707187,  0.49418585,
       -0.48207441,  0.25288888,  0.10722764,  2.4330126 ,  0.55108954,
       -0.73962193, -1.33632173,  1.177087  ,  0.46233501, -0.97847408,
        0.09680973, -0.03848715,  1.5971585 ,  0.47956492,  1.79283347,
        1.32710166,  2.38450083,  2.9369401 ,  2.14681113,  2.36986949,
        3.06384157,  3.91575378,  3.93646339,  3.09427612,  2.37447163,
        2.77881295,  2.28656128,  2.98563349,  2.3751947 ,  2.20986553,
        2.625621  ,  4.28063878,  3.58264137,  2.80706372,  2.89965933,
        2.32073698,  2.54983095,  1.81254128,  2.76014464,  2.7371505 ,
        3.60486887,  2.889826  ,  3.39215608,  1.0481819 ,  1.60991228,
        3.14313097,  2.2401569 ,  2.84767378,  2.59749706,  2.94929937,
        3.53003227,  2.40611054,  2.92908473,  2.18141278,  2.38092779,
        3.21161722,  3.67791872,  2.4655558 ,  3.37052415,  2.60195585,
        2.67783946,  2.38701709,  3.20875816])

```

我们看到它们确实一致。

第一个主成分对 V8（-0.423）、V7（-0.395）、V13（-0.376）、V10（-0.313）、V12（-0.297）、V14（-0.287）、V9（0.299）、V3（0.245）和 V5（0.239）具有最高（绝对值）的载荷。V8、V7、V13、V10、V12 和 V14 的载荷为负，而 V9、V3 和 V5 的载荷为正。因此，第一个主成分的解释是它代表了 V8、V7、V13、V10、V12 和 V14 的浓度与 V9、V3 和 V5 的浓度之间的对比。

类似地，我们可以通过输入以下内容来获取第二个主成分的载荷：

```
pca.components_[1]

```

```
array([ 0.48365155,  0.22493093,  0.31606881, -0.0105905 ,  0.299634  ,
        0.06503951, -0.00335981,  0.02877949,  0.03930172,  0.52999567,
       -0.27923515, -0.16449619,  0.36490283])

```

这意味着第二个主成分是变量的线性组合：

```
0.484*Z2 + 0.225*Z3 + 0.316*Z4 - 0.011*Z5 + 0.300*Z6 + 0.065*Z7 - 0.003*Z8 + 0.029*Z9 + 0.039*Z10 + 0.530*Z11 - 0.279*Z12 - 0.164*Z13 + 0.365*Z14

```

其中 Z1、Z2、Z3，...，Z14 是变量 V2、V3，...，V14 的标准化版本，每个版本的均值为 0，方差为 1。

请注意，载荷的平方和为 1，如上所示：

```
np.sum(pca.components_[1]**2)

```

```
1.0000000000000011

```

第二主成分对 V11（0.530）、V2（0.484）、V14（0.365）、V4（0.316）、V6（0.300）、V12（-0.279）和 V3（0.225）具有最高的载荷。V11、V2、V14、V4、V6 和 V3 的载荷为正，而 V12 的载荷为负。因此，第二主成分的解释是它代表了 V11、V2、V14、V4、V6 和 V3 的浓度与 V12 的浓度之间的对比。请注意，V11（0.530）和 V2（0.484）的载荷最大，因此对比主要在 V11 和 V2 的浓度之间，以及 V12 的浓度之间。### 主成分的散点图

主成分的值可以通过`PCA`类的`transform()`（或`fit_transform()`）方法计算。它返回一个包含主成分的矩阵，其中矩阵的第一列包含第一个主成分，第二列包含第二个主成分，依此类推。

因此，在我们的示例中，`pca.transform(standardisedX)[:, 0]`包含第一个主成分，`pca.transform(standardisedX)[:, 1]`包含第二个主成分。

我们可以制作前两个主成分的散点图，并通过输入以下内容为葡萄酒样本的数据点标记出其来自的品种：

```
def pca_scatter(pca, standardised_values, classifs):
    foo = pca.transform(standardised_values)
    bar = pd.DataFrame(zip(foo[:, 0], foo[:, 1], classifs), columns=["PC1", "PC2", "Class"])
    sns.lmplot("PC1", "PC2", bar, hue="Class", fit_reg=False)

pca_scatter(pca, standardisedX, y)

```

![png](img/a_little_book_of_python_for_multivariate_analysis_112_0.png)

散点图将第一个主成分显示在 x 轴上，将第二个主成分显示在 y 轴上。从散点图中我们可以看到，品种 1 的葡萄酒样本的第一个主成分值要低得多，而品种 3 的葡萄酒样本的第一个主成分值要高。因此，第一个主成分将品种 1 的葡萄酒样本与品种 3 的葡萄酒样本分开。

我们还可以看到，品种 2 的葡萄酒样本的第二主成分值比品种 1 和 3 的葡萄酒样本要高得多。因此，第二主成分将品种 2 的样本与品种 1 和 3 的样本分开。

因此，前两个主成分对于区分三种不同品种的葡萄酒样本是相当有用的。

在上面，我们将第一个主成分解释为 V8、V7、V13、V10、V12 和 V14 的浓度与 V9、V3 和 V5 的浓度之间的对比。我们可以通过使用`printMeanAndSdByGroup()`函数（见上文）打印出每个品种中标准化浓度变量的均值，以检查这在不同品种中的化学物质浓度方面是否合理：

```
printMeanAndSdByGroup(standardisedX, y);

```

```
## Means:

```

|  | V2 | V3 | V4 | V5 | V6 | V7 | V8 | V9 | V10 | V11 | V12 | V13 | V14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V1 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.919195 | -0.292342 | 0.325604 | -0.737997 | 0.463226 | 0.873362 | 0.956884 | -0.578985 | 0.540383 | 0.203401 | 0.458847 | 0.771351 | 1.174501 |
| 2 | -0.891720 | -0.362362 | -0.444958 | 0.223137 | -0.364567 | -0.058067 | 0.051780 | 0.014569 | 0.069002 | -0.852799 | 0.433611 | 0.245294 | -0.724110 |
| 3 | 0.189159 | 0.895331 | 0.257945 | 0.577065 | -0.030127 | -0.987617 | -1.252761 | 0.690119 | -0.766287 | 1.011418 | -1.205382 | -1.310950 | -0.372578 |

```
## Standard deviations:

```

|  | V2 | V3 | V4 | V5 | V6 | V7 | V8 | V9 | V10 | V11 | V12 | V13 | V14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V1 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.565989 | 0.612825 | 0.823302 | 0.758115 | 0.730892 | 0.538506 | 0.395674 | 0.559639 | 0.715905 | 0.531210 | 0.506699 | 0.500058 | 0.699428 |
| 2 | 0.659832 | 0.905196 | 1.144991 | 0.998777 | 1.168006 | 0.867674 | 0.703493 | 0.991797 | 1.047418 | 0.397269 | 0.884060 | 0.696425 | 0.497100 |
| 3 | 0.648130 | 0.966347 | 0.668036 | 0.670991 | 0.756649 | 0.565996 | 0.291583 | 0.989818 | 0.708814 | 0.989176 | 0.496834 | 0.380317 | 0.362688 |

```
## Sample sizes:

```

|  | 0 |
| --- | --- |
| V1 |  |
| --- | --- |
| 1 | 59 |
| 2 | 71 |
| 3 | 48 |

第一主成分能够将品种 1 与品种 3 区分开，这是否合理？在品种 1 中，V8（0.954）、V7（0.871）、V13（0.769）、V10（0.539）、V12（0.458）和 V14（1.171）的均值相对于 V9（-0.577）、V3（-0.292）和 V5（-0.736）的均值非常高。在品种 3 中，V8（-1.249）、V7（-0.985）、V13（-1.307）、V10（-0.764）、V12（-1.202）和 V14（-0.372）的均值相对于 V9（0.688）、V3（0.893）和 V5（0.575）的均值非常低。因此，第一主成分确实是 V8、V7、V13、V10、V12 和 V14 的浓度与 V9、V3 和 V5 的浓度之间的对比；第一主成分能够将品种 1 与品种 3 区分开。

在上面，我们解释了第二主成分作为 V11、V2、V14、V4、V6 和 V3 的浓度与 V12 浓度之间的对比。根据这些变量在不同品种中的均值，在第二主成分能够将品种 2 与品种 1 和品种 3 区分开是有意义的吗？在品种 1 中，V11（0.203）、V2（0.917）、V14（1.171）、V4（0.325）、V6（0.462）和 V3（-0.292）的均值与 V12（0.458）的均值并没有太大差异。在品种 3 中，V11（1.009）、V2（0.189）、V14（-0.372）、V4（0.257）、V6（-0.030）和 V3（0.893）的均值也与 V12（-1.202）的均值并没有太大差异。相反，在品种 2 中，V11（-0.850）、V2（-0.889）、V14（-0.722）、V4（-0.444）、V6（-0.364）和 V3（-0.361）的均值要远远小于 V12（0.432）的均值。因此，主成分作为 V11、V2、V14、V4、V6 和 V3 的浓度与 V12 浓度之间的对比是有意义的；主成分 2 能够将品种 2 与品种 1 和品种 3 区分开。## 线性判别分析

主成分分析的目的是找到多元数据集变化的最佳低维表示。例如，在葡萄酒数据集中，我们有 13 种化学浓度描述来自三个品种的葡萄酒样本。通过进行主成分分析，我们发现化学浓度样本之间的大部分变化可以用前两个主成分来描述，其中每个主成分是 13 种化学浓度的特定线性组合。

线性判别分析（LDA）的目的是找到原始变量（这里是 13 种化学浓度）的线性组合，以在数据集中实现群组（这里是葡萄酒品种）之间的最佳分离。*线性判别分析* 也称为*典型判别分析*，或简称*判别分析*。

如果我们想按品种分开葡萄酒，那么葡萄酒来自三个不同的品种，所以组数（G）为 3，变量数为 13（13 种化学浓度；p = 13）。能够通过品种将葡萄酒分开的有用判别函数的最大数量是 G-1 和 p 的最小值，在这种情况下是 2 和 13 的最小值，即 2。因此，我们最多可以找到 2 个有用的判别函数，使用 13 种化学浓度变量来按品种分开葡萄酒。

您可以通过使用 `sklearn.discriminant_analysis` 模块的 `LinearDiscriminantAnalysis` 类模型，并使用其 `fit()` 方法来适应我们的 `X, y` 数据来进行线性判别分析。

举例来说，要使用葡萄酒样本中的 13 种化学浓度进行线性判别分析，我们输入以下命令：

```
lda = LinearDiscriminantAnalysis().fit(X, y)

```

### 判别函数的载荷

葡萄酒数据的判别函数的载荷值存储在 `lda` 对象模型的 `scalings_` 成员中。为了漂亮打印，我们可以键入：

```
def pretty_scalings(lda, X, out=False):
    ret = pd.DataFrame(lda.scalings_, index=X.columns, columns=["LD"+str(i+1) for i in range(lda.scalings_.shape[1])])
    if out:
        print("Coefficients of linear discriminants:")
        display(ret)
    return ret

pretty_scalings_ = pretty_scalings(lda, X, out=True)

```

```
Coefficients of linear discriminants:

```

|  | LD1 | LD2 |
| --- | --- | --- |
| V2 | -0.403400 | 0.871793 |
| V3 | 0.165255 | 0.305380 |
| V4 | -0.369075 | 2.345850 |
| V5 | 0.154798 | -0.146381 |
| V6 | -0.002163 | -0.000463 |
| V7 | 0.618052 | -0.032213 |
| V8 | -1.661191 | -0.491998 |
| V9 | -1.495818 | -1.630954 |
| V10 | 0.134093 | -0.307088 |
| V11 | 0.355056 | 0.253231 |
| V12 | -0.818036 | -1.515634 |
| V13 | -1.157559 | 0.051184 |
| V14 | -0.002691 | 0.002853 |

这意味着第一个判别函数是变量的线性组合：

```
-0.403*V2 + 0.165*V3 - 0.369*V4 + 0.155*V5 - 0.002*V6 + 0.618*V7 - 1.661*V8 - 1.496*V9 + 0.134*V10 + 0.355*V11 - 0.818*V12 - 1.158*V13 - 0.003*V14

```

其中 V2、V3、...、V14 是葡萄酒样本中发现的 14 种化学物质的浓度。为方便起见，每个判别函数（例如第一个判别函数）的值都经过缩放，使其平均值为零（见下文）。

请注意，这些载荷是这样计算的，以使每个组（品种）的每个判别函数的组内方差等于 1，如下所示。

如上所述，这些缩放值存储在由 `LinearDiscriminantAnalysis().fit(X, y)` 返回的对象变量的命名成员 `scalings_` 中。该元素包含一个 numpy 数组，其中第一列包含第一个判别函数的载荷，第二列包含第二个判别函数的载荷，依此类推。例如，要提取第一个判别函数的载荷，我们可以键入：

```
lda.scalings_[:, 0]

```

```
array([-0.40339978,  0.1652546 , -0.36907526,  0.15479789, -0.0021635 ,
        0.61805207, -1.66119123, -1.49581844,  0.13409263,  0.35505571,
       -0.81803607, -1.15755938, -0.00269121])

```

或者为了“更漂亮”的打印，使用上面创建的 dataframe 变量：

```
pretty_scalings_.LD1

```

```
V2    -0.403400
V3     0.165255
V4    -0.369075
V5     0.154798
V6    -0.002163
V7     0.618052
V8    -1.661191
V9    -1.495818
V10    0.134093
V11    0.355056
V12   -0.818036
V13   -1.157559
V14   -0.002691
Name: LD1, dtype: float64

```

要计算第一个判别函数的值，我们可以定义自己的函数 `calclda()`：

```
def calclda(variables, loadings):
    # find the number of samples in the data set and the number of variables
    numsamples, numvariables = variables.shape
    # make a vector to store the discriminant function
    ld = np.zeros(numsamples)
    # calculate the value of the discriminant function for each sample
    for i in range(numsamples):
        valuei = 0
        for j in range(numvariables):
            valueij = variables.iloc[i, j]
            loadingj = loadings[j]
            valuei = valuei + (valueij * loadingj)
        ld[i] = valuei
    # standardise the discriminant function so that its mean value is 0:
    ld = scale(ld, with_std=False)
    return ld

```

函数 `calclda()` 简单地计算数据集中每个样本的判别函数的值，例如，对于第一个判别函数，对于每个样本，我们使用以下方程计算值：

```
-0.403*V2 - 0.165*V3 - 0.369*V4 + 0.155*V5 - 0.002*V6 + 0.618*V7 - 1.661*V8 - 1.496*V9 + 0.134*V10 + 0.355*V11 - 0.818*V12 - 1.158*V13 - 0.003*V14

```

此外，`calclda()` 函数内部使用 `scale()` 命令来标准化判别函数的值（例如第一个判别函数），使其均值（在所有葡萄酒样本上）为 0。

我们可以使用函数 `calclda()` 来计算我们的葡萄酒数据中每个样本的第一个判别函数的值：

```
calclda(X, lda.scalings_[:, 0])

```

```
array([-4.70024401, -4.30195811, -3.42071952, -4.20575366, -1.50998168,
       -4.51868934, -4.52737794, -4.14834781, -3.86082876, -3.36662444,
       -4.80587907, -3.42807646, -3.66610246, -5.58824635, -5.50131449,
       -3.18475189, -3.28936988, -2.99809262, -5.24640372, -3.13653106,
       -3.57747791, -1.69077135, -4.83515033, -3.09588961, -3.32164716,
       -2.14482223, -3.9824285 , -2.68591432, -3.56309464, -3.17301573,
       -2.99626797, -3.56866244, -3.38506383, -3.5275375 , -2.85190852,
       -2.79411996, -2.75808511, -2.17734477, -3.02926382, -3.27105228,
       -2.92065533, -2.23721062, -4.69972568, -1.23036133, -2.58203904,
       -2.58312049, -3.88887889, -3.44975356, -2.34223331, -3.52062596,
       -3.21840912, -4.38214896, -4.36311727, -3.51917293, -3.12277475,
       -1.8024054 , -2.87378754, -3.61690518, -3.73868551,  1.58618749,
        0.79967216,  2.38015446, -0.45917726, -0.50726885,  0.39398359,
       -0.92256616, -1.95549377, -0.34732815,  0.20371212, -0.24831914,
        1.17987999, -1.07718925,  0.64100179, -1.74684421, -0.34721117,
        1.14274222,  0.18665882,  0.900525  , -0.70709551, -0.59562833,
       -0.55761818, -1.80430417,  0.23077079,  2.03482711, -0.62113021,
       -1.03372742,  0.76598781,  0.35042568,  0.15324508, -0.14962842,
        0.48079504,  1.39689016,  0.91972331, -0.59102937,  0.49411386,
       -1.62614426,  2.00044562, -1.00534818, -2.07121314, -1.6381589 ,
       -1.0589434 ,  0.02594549, -0.21887407,  1.3643764 , -1.12901245,
       -0.21263094, -0.77946884,  0.61546732,  0.22550192, -2.03869851,
        0.79274716,  0.30229545, -0.50664882,  0.99837397, -0.21954922,
       -0.37131517,  0.05545894, -0.09137874,  1.79755252, -0.17405009,
       -1.17870281, -3.2105439 ,  0.62605202,  0.03366613, -0.6993008 ,
       -0.72061079, -0.51933512,  1.17030045,  0.10824791,  1.12319783,
        2.24632419,  3.28527755,  4.07236441,  3.86691235,  3.45088333,
        3.71583899,  3.9222051 ,  4.8516102 ,  3.54993389,  3.76889174,
        2.6694225 ,  2.32491492,  3.17712883,  2.88964418,  3.78325562,
        3.04411324,  4.70697017,  4.85021393,  4.98359184,  4.86968293,
        4.5986919 ,  5.67447884,  5.32986123,  5.03401031,  4.52080087,
        5.0978371 ,  5.04368277,  4.86980829,  5.61316558,  5.67046737,
        5.37413513,  3.09975377,  3.35888137,  3.04007194,  4.94861303,
        4.54504458,  5.27255844,  5.13016117,  4.30468082,  5.08336782,
        4.06743571,  5.74212961,  4.4820514 ,  4.29150758,  4.50329623,
        5.04747033,  4.27615505,  5.5380861 ])

```

实际上，可以使用 LDA 对象的 `transform(X)` 或 `fit_transform(X, y)` 方法来计算第一个线性判别函数的值，因此我们可以将其与我们计算的值进行比较，它们应该是一致的：

```
# Try either, they produce the same result, use help() for more info
# lda.transform(X)[:, 0]
lda.fit_transform(X, y)[:, 0]

```

```
array([-4.70024401, -4.30195811, -3.42071952, -4.20575366, -1.50998168,
       -4.51868934, -4.52737794, -4.14834781, -3.86082876, -3.36662444,
       -4.80587907, -3.42807646, -3.66610246, -5.58824635, -5.50131449,
       -3.18475189, -3.28936988, -2.99809262, -5.24640372, -3.13653106,
       -3.57747791, -1.69077135, -4.83515033, -3.09588961, -3.32164716,
       -2.14482223, -3.9824285 , -2.68591432, -3.56309464, -3.17301573,
       -2.99626797, -3.56866244, -3.38506383, -3.5275375 , -2.85190852,
       -2.79411996, -2.75808511, -2.17734477, -3.02926382, -3.27105228,
       -2.92065533, -2.23721062, -4.69972568, -1.23036133, -2.58203904,
       -2.58312049, -3.88887889, -3.44975356, -2.34223331, -3.52062596,
       -3.21840912, -4.38214896, -4.36311727, -3.51917293, -3.12277475,
       -1.8024054 , -2.87378754, -3.61690518, -3.73868551,  1.58618749,
        0.79967216,  2.38015446, -0.45917726, -0.50726885,  0.39398359,
       -0.92256616, -1.95549377, -0.34732815,  0.20371212, -0.24831914,
        1.17987999, -1.07718925,  0.64100179, -1.74684421, -0.34721117,
        1.14274222,  0.18665882,  0.900525  , -0.70709551, -0.59562833,
       -0.55761818, -1.80430417,  0.23077079,  2.03482711, -0.62113021,
       -1.03372742,  0.76598781,  0.35042568,  0.15324508, -0.14962842,
        0.48079504,  1.39689016,  0.91972331, -0.59102937,  0.49411386,
       -1.62614426,  2.00044562, -1.00534818, -2.07121314, -1.6381589 ,
       -1.0589434 ,  0.02594549, -0.21887407,  1.3643764 , -1.12901245,
       -0.21263094, -0.77946884,  0.61546732,  0.22550192, -2.03869851,
        0.79274716,  0.30229545, -0.50664882,  0.99837397, -0.21954922,
       -0.37131517,  0.05545894, -0.09137874,  1.79755252, -0.17405009,
       -1.17870281, -3.2105439 ,  0.62605202,  0.03366613, -0.6993008 ,
       -0.72061079, -0.51933512,  1.17030045,  0.10824791,  1.12319783,
        2.24632419,  3.28527755,  4.07236441,  3.86691235,  3.45088333,
        3.71583899,  3.9222051 ,  4.8516102 ,  3.54993389,  3.76889174,
        2.6694225 ,  2.32491492,  3.17712883,  2.88964418,  3.78325562,
        3.04411324,  4.70697017,  4.85021393,  4.98359184,  4.86968293,
        4.5986919 ,  5.67447884,  5.32986123,  5.03401031,  4.52080087,
        5.0978371 ,  5.04368277,  4.86980829,  5.61316558,  5.67046737,
        5.37413513,  3.09975377,  3.35888137,  3.04007194,  4.94861303,
        4.54504458,  5.27255844,  5.13016117,  4.30468082,  5.08336782,
        4.06743571,  5.74212961,  4.4820514 ,  4.29150758,  4.50329623,
        5.04747033,  4.27615505,  5.5380861 ])

```

我们看到它们确实是一致的。

与主成分分析中需要标准化输入变量不同，线性判别分析中的输入变量是否标准化无关紧要。然而，在线性判别分析中使用标准化的变量使得更容易解释判别函数中的载荷。

在线性判别分析中，输入变量的标准化版本被定义为具有零均值和组内方差为 1。 因此，我们可以通过从变量的每个值中减去均值，并除以组内标准差来计算“组标准化”变量。 要计算一组变量的组标准化版本，我们可以使用下面的`groupStandardise()`函数：

```
def groupStandardise(variables, groupvariable):
    # find the number of samples in the data set and the number of variables
    numsamples, numvariables = variables.shape
    # find the variable names
    variablenames = variables.columns
    # calculate the group-standardised version of each variable
    variables_new = pd.DataFrame()
    for i in range(numvariables):
        variable_name = variablenames[i]
        variablei = variables[variable_name]
        variablei_Vw = calcWithinGroupsVariance(variablei, groupvariable)
        variablei_mean = np.mean(variablei)
        variablei_new = (variablei - variablei_mean)/(np.sqrt(variablei_Vw))
        variables_new[variable_name] = variablei_new
    return variables_new

```

例如，我们可以使用`groupStandardise()`函数来计算酒样品中化学浓度的组标准化版本：

```
groupstandardisedX = groupStandardise(X, y)

```

我们随后可以使用`LinearDiscriminantAnalysis().fit()`方法对组标准化变量进行线性判别分析：

```
lda2 = LinearDiscriminantAnalysis().fit(groupstandardisedX, y)
pretty_scalings(lda2, groupstandardisedX)

```

|  | LD1 | LD2 |
| --- | --- | --- |
| V2 | -0.206505 | 0.446280 |
| V3 | 0.155686 | 0.287697 |
| V4 | -0.094869 | 0.602989 |
| V5 | 0.438021 | -0.414204 |
| V6 | -0.029079 | -0.006220 |
| V7 | 0.270302 | -0.014088 |
| V8 | -0.870673 | -0.257869 |
| V9 | -0.163255 | -0.178004 |
| V10 | 0.066531 | -0.152364 |
| V11 | 0.536701 | 0.382783 |
| V12 | -0.128011 | -0.237175 |
| V13 | -0.464149 | 0.020523 |
| V14 | -0.463854 | 0.491738 |

用组标准化变量解释计算出的载荷而不是原始（未标准化）变量的载荷是有意义的。

在为组标准化变量计算的第一个判别函数中，绝对值最大的载荷分别为 V8（-0.871）、V11（0.537）、V13（-0.464）、V14（-0.464）和 V5（0.438）。 V8、V13 和 V14 的载荷为负，而 V11 和 V5 的载荷为正。 因此，判别函数似乎代表了 V8、V13 和 V14 的浓度与 V11 和 V5 的浓度之间的对比。

我们在上面看到，给出各组之间最大分离的个体变量是 V8（分离度 233.93）、V14（207.92）、V13（189.97）、V2（135.08）和 V11（120.66）。 这些大部分是在线性判别函数中具有最大载荷的变量（V8 的载荷：-0.871，V14 的载荷：-0.464，V13 的载荷：-0.464，V11 的载荷：0.537）。

我们发现上面的变量 V8 和 V11 具有负的组间协方差（-60.41）和正的组内协方差（0.29）。 当两个变量的组间协方差和组内协方差的符号相反时，表明通过使用这两个变量的线性组合可以获得比仅使用其中一个变量更好的组别分离。

因此，考虑到两个变量 V8 和 V11 的组间和组内协方差符号相反，并且这两个变量在单独使用时给出了最大的组别分离，因此并不奇怪这两个变量在第一个判别函数中具有最大的载荷。

请注意，尽管组标准化变量的载荷比未标准化变量的载荷更容易解释，但判别函数的值不受输入变量是否标准化的影响是相同的。例如，对于酒数据，我们可以计算使用未标准化和组标准化变量计算的第一个判别函数的值，方法是键入：

```
lda.fit_transform(X, y)[:, 0]

```

```
array([-4.70024401, -4.30195811, -3.42071952, -4.20575366, -1.50998168,
       -4.51868934, -4.52737794, -4.14834781, -3.86082876, -3.36662444,
       -4.80587907, -3.42807646, -3.66610246, -5.58824635, -5.50131449,
       -3.18475189, -3.28936988, -2.99809262, -5.24640372, -3.13653106,
       -3.57747791, -1.69077135, -4.83515033, -3.09588961, -3.32164716,
       -2.14482223, -3.9824285 , -2.68591432, -3.56309464, -3.17301573,
       -2.99626797, -3.56866244, -3.38506383, -3.5275375 , -2.85190852,
       -2.79411996, -2.75808511, -2.17734477, -3.02926382, -3.27105228,
       -2.92065533, -2.23721062, -4.69972568, -1.23036133, -2.58203904,
       -2.58312049, -3.88887889, -3.44975356, -2.34223331, -3.52062596,
       -3.21840912, -4.38214896, -4.36311727, -3.51917293, -3.12277475,
       -1.8024054 , -2.87378754, -3.61690518, -3.73868551,  1.58618749,
        0.79967216,  2.38015446, -0.45917726, -0.50726885,  0.39398359,
       -0.92256616, -1.95549377, -0.34732815,  0.20371212, -0.24831914,
        1.17987999, -1.07718925,  0.64100179, -1.74684421, -0.34721117,
        1.14274222,  0.18665882,  0.900525  , -0.70709551, -0.59562833,
       -0.55761818, -1.80430417,  0.23077079,  2.03482711, -0.62113021,
       -1.03372742,  0.76598781,  0.35042568,  0.15324508, -0.14962842,
        0.48079504,  1.39689016,  0.91972331, -0.59102937,  0.49411386,
       -1.62614426,  2.00044562, -1.00534818, -2.07121314, -1.6381589 ,
       -1.0589434 ,  0.02594549, -0.21887407,  1.3643764 , -1.12901245,
       -0.21263094, -0.77946884,  0.61546732,  0.22550192, -2.03869851,
        0.79274716,  0.30229545, -0.50664882,  0.99837397, -0.21954922,
       -0.37131517,  0.05545894, -0.09137874,  1.79755252, -0.17405009,
       -1.17870281, -3.2105439 ,  0.62605202,  0.03366613, -0.6993008 ,
       -0.72061079, -0.51933512,  1.17030045,  0.10824791,  1.12319783,
        2.24632419,  3.28527755,  4.07236441,  3.86691235,  3.45088333,
        3.71583899,  3.9222051 ,  4.8516102 ,  3.54993389,  3.76889174,
        2.6694225 ,  2.32491492,  3.17712883,  2.88964418,  3.78325562,
        3.04411324,  4.70697017,  4.85021393,  4.98359184,  4.86968293,
        4.5986919 ,  5.67447884,  5.32986123,  5.03401031,  4.52080087,
        5.0978371 ,  5.04368277,  4.86980829,  5.61316558,  5.67046737,
        5.37413513,  3.09975377,  3.35888137,  3.04007194,  4.94861303,
        4.54504458,  5.27255844,  5.13016117,  4.30468082,  5.08336782,
        4.06743571,  5.74212961,  4.4820514 ,  4.29150758,  4.50329623,
        5.04747033,  4.27615505,  5.5380861 ])

```

```
lda2.fit_transform(groupstandardisedX, y)[:, 0]

```

```
array([-4.70024401, -4.30195811, -3.42071952, -4.20575366, -1.50998168,
       -4.51868934, -4.52737794, -4.14834781, -3.86082876, -3.36662444,
       -4.80587907, -3.42807646, -3.66610246, -5.58824635, -5.50131449,
       -3.18475189, -3.28936988, -2.99809262, -5.24640372, -3.13653106,
       -3.57747791, -1.69077135, -4.83515033, -3.09588961, -3.32164716,
       -2.14482223, -3.9824285 , -2.68591432, -3.56309464, -3.17301573,
       -2.99626797, -3.56866244, -3.38506383, -3.5275375 , -2.85190852,
       -2.79411996, -2.75808511, -2.17734477, -3.02926382, -3.27105228,
       -2.92065533, -2.23721062, -4.69972568, -1.23036133, -2.58203904,
       -2.58312049, -3.88887889, -3.44975356, -2.34223331, -3.52062596,
       -3.21840912, -4.38214896, -4.36311727, -3.51917293, -3.12277475,
       -1.8024054 , -2.87378754, -3.61690518, -3.73868551,  1.58618749,
        0.79967216,  2.38015446, -0.45917726, -0.50726885,  0.39398359,
       -0.92256616, -1.95549377, -0.34732815,  0.20371212, -0.24831914,
        1.17987999, -1.07718925,  0.64100179, -1.74684421, -0.34721117,
        1.14274222,  0.18665882,  0.900525  , -0.70709551, -0.59562833,
       -0.55761818, -1.80430417,  0.23077079,  2.03482711, -0.62113021,
       -1.03372742,  0.76598781,  0.35042568,  0.15324508, -0.14962842,
        0.48079504,  1.39689016,  0.91972331, -0.59102937,  0.49411386,
       -1.62614426,  2.00044562, -1.00534818, -2.07121314, -1.6381589 ,
       -1.0589434 ,  0.02594549, -0.21887407,  1.3643764 , -1.12901245,
       -0.21263094, -0.77946884,  0.61546732,  0.22550192, -2.03869851,
        0.79274716,  0.30229545, -0.50664882,  0.99837397, -0.21954922,
       -0.37131517,  0.05545894, -0.09137874,  1.79755252, -0.17405009,
       -1.17870281, -3.2105439 ,  0.62605202,  0.03366613, -0.6993008 ,
       -0.72061079, -0.51933512,  1.17030045,  0.10824791,  1.12319783,
        2.24632419,  3.28527755,  4.07236441,  3.86691235,  3.45088333,
        3.71583899,  3.9222051 ,  4.8516102 ,  3.54993389,  3.76889174,
        2.6694225 ,  2.32491492,  3.17712883,  2.88964418,  3.78325562,
        3.04411324,  4.70697017,  4.85021393,  4.98359184,  4.86968293,
        4.5986919 ,  5.67447884,  5.32986123,  5.03401031,  4.52080087,
        5.0978371 ,  5.04368277,  4.86980829,  5.61316558,  5.67046737,
        5.37413513,  3.09975377,  3.35888137,  3.04007194,  4.94861303,
        4.54504458,  5.27255844,  5.13016117,  4.30468082,  5.08336782,
        4.06743571,  5.74212961,  4.4820514 ,  4.29150758,  4.50329623,
        5.04747033,  4.27615505,  5.5380861 ])

```

我们可以看到，虽然使用未标准化数据和组标准化数据计算出的第一判别函数的载荷不同，但第一判别函数的实际值是相同的。### 判别函数达到的分离度

要计算每个判别函数实现的分离度，我们首先需要计算每个判别函数的值，方法是将变量的值代入判别函数的线性组合中（例如，对于第一个判别函数，`-0.403*V2 - 0.165*V3 - 0.369*V4 + 0.155*V5 - 0.002*V6 + 0.618*V7 - 1.661*V8 - 1.496*V9 + 0.134*V10 + 0.355*V11 - 0.818*V12 - 1.158*V13 - 0.003*V14`），然后缩放判别函数的值，使它们的平均值为零。

如上所述，我们可以使用`rpredict()`函数来模拟 R 中`predict()`函数的输出。例如，要计算酒数据的判别函数的值，我们键入：

```
def rpredict(lda, X, y, out=False):
    ret = {"class": lda.predict(X),
           "posterior": pd.DataFrame(lda.predict_proba(X), columns=lda.classes_)}
    ret["x"] = pd.DataFrame(lda.fit_transform(X, y))
    ret["x"].columns = ["LD"+str(i+1) for i in range(ret["x"].shape[1])]
    if out:
        print("class")
        print(ret["class"])
        print()
        print("posterior")
        print(ret["posterior"])
        print()
        print("x")
        print(ret["x"])
    return ret

lda_values = rpredict(lda, standardisedX, y, True)

```

```
class
['2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2'
 '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2'
 '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2'
 '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2'
 '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2'
 '2' '2' '2' '2' '2' '2' '3' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2'
 '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2'
 '2' '2' '2' '2' '3' '3' '3' '3' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2'
 '3' '2' '3' '3' '3' '3' '3' '3' '3' '2' '2' '2' '3' '2' '3' '3' '2' '2'
 '2' '2' '3' '2' '3' '3' '3' '3' '2' '2' '3' '3' '3' '3' '2' '3']

posterior
                1         2         3
0    1.344367e-22  0.999236  0.000764
1    4.489007e-27  0.983392  0.016608
2    2.228888e-24  0.791616  0.208384
3    1.026755e-24  0.500161  0.499839
4    6.371860e-23  0.790657  0.209343
5    1.552082e-24  0.981986  0.018014
6    3.354960e-23  0.951823  0.048177
7    3.417899e-22  0.925154  0.074846
8    4.041139e-26  0.978998  0.021002
9    3.718868e-26  0.619841  0.380159
..            ...       ...       ...
168  7.463695e-30  0.500000  0.500000
169  1.389203e-29  0.499927  0.500073
170  1.356187e-33  0.500000  0.500000
171  1.007615e-33  0.500000  0.500000
172  1.524219e-30  0.500000  0.500000
173  1.317492e-30  0.500000  0.500000
174  2.664128e-32  0.500000  0.500000
175  2.873436e-34  0.500000  0.500000
176  1.479166e-32  0.500000  0.500000
177  1.209888e-28  0.500000  0.500000

[178 rows x 3 columns]

x
          LD1       LD2
0   -4.700244  1.979138
1   -4.301958  1.170413
2   -3.420720  1.429101
3   -4.205754  4.002871
4   -1.509982  0.451224
5   -4.518689  3.213138
6   -4.527378  3.269122
7   -4.148348  3.104118
8   -3.860829  1.953383
9   -3.366624  1.678643
..        ...       ...
168  4.304681  2.391125
169  5.083368  3.157667
170  4.067436  0.318922
171  5.742130  1.467082
172  4.482051  3.307084
173  4.291508  3.390332
174  4.503296  2.083546
175  5.047470  3.196231
176  4.276155  2.431388
177  5.538086  3.042057

[178 rows x 2 columns]

```

返回的变量具有一个命名元素`x`，它是包含线性判别函数的矩阵：`x`的第一列包含第一个判别函数，`x`的第二列包含第二个判别函数，依此类推（如果有更多判别函数）。

因此，我们可以使用`calcSeparations()`函数（见上文）来计算酒数据的两个线性判别函数所达到的分离度，该函数将分离度计算为组间方差与组内方差之比：

```
calcSeparations(lda_values["x"], y)

```

```
variable LD1 Vw= 1.0 Vb= 794.652200566 separation= 794.652200566
variable LD2 Vw= 1.0 Vb= 361.241041493 separation= 361.241041493

```

如上所述，每个判别函数的载荷是这样计算的，即每个组（这里是酒的品种）的组内方差（`Vw`）等于 1，正如我们在上面从`calcSeparations()`的输出中看到的。

`calcSeparations()`的输出告诉我们，第一个（最佳的）判别函数实现的分离度为 794.7，第二个（次佳的）判别函数实现的分离度为 361.2。

因此，总分离度是这些分离度的总和，即（`794.652200566216+361.241041493455=1155.893`）1155.89，保留两位小数。因此，第一个判别函数实现的*百分比分离度*是（`794.652200566216*100/1155.893=`）68.75%，而第二个判别函数实现的百分比分离度是（`361.241041493455*100/1155.893=`）31.25%。

*迹的比例*（在 R 中由`lda()`模型报告）是每个判别函数实现的百分比分离。例如，对于葡萄酒数据，我们得到与刚刚计算的相同的值（68.75%和 31.25%）。请注意，在`sklearn`中，迹的比例报告为`LinearDiscriminantAnalysis`模型中的`explained_variance_ratio_`，仅对“eigen”求解器计算，而到目前为止，我们一直在使用默认求解器，即“svd”（奇异值分解）：

```
def proportion_of_trace(lda):
    ret = pd.DataFrame([round(i, 4) for i in lda.explained_variance_ratio_ if round(i, 4) > 0], columns=["ExplainedVariance"])
    ret.index = ["LD"+str(i+1) for i in range(ret.shape[0])]
    ret = ret.transpose()
    print("Proportion of trace:")
    print(ret.to_string(index=False))
    return ret

proportion_of_trace(LinearDiscriminantAnalysis(solver="eigen").fit(X, y));

```

```
Proportion of trace:
    LD1     LD2
 0.6875  0.3125

```

因此，第一个判别函数确实在三个组（三个品种）之间实现了很好的分离，但第二个判别函数在很大程度上提高了组的分离程度，因此是否值得也使用第二个判别函数。因此，为了实现组（品种）的良好分离，有必要使用前两个判别函数。

我们之前发现，任何单个变量（单个化学浓度）的最大分离值为 233.9，远远小于 794.7，第一个判别函数实现的分离值。因此，使用多个变量来计算判别函数的效果是，我们可以找到一个判别函数，它在组之间的分离要比任何单个变量单独使用的分离要大得多。### 线性判别分析（LDA）值的堆积直方图

显示线性判别分析（LDA）结果的一种好方法是制作不同组样本（我们示例中的不同葡萄酒品种）的判别函数值的堆积直方图。

我们可以使用下面定义的`ldahist()`函数来做到这一点。

```
def ldahist(data, g, sep=False):
    xmin = np.trunc(np.min(data)) - 1
    xmax = np.trunc(np.max(data)) + 1
    ncol = len(set(g))
    binwidth = 0.5
    bins=np.arange(xmin, xmax + binwidth, binwidth)
    if sep:
        fig, axl = plt.subplots(ncol, 1, sharey=True, sharex=True)
    else:
        fig, axl = plt.subplots(1, 1, sharey=True, sharex=True)
        axl = [axl]*ncol
    for ax, (group, gdata) in zip(axl, data.groupby(g)):
        sns.distplot(gdata.values, bins, ax=ax, label="group "+str(group))
        ax.set_xlim([xmin, xmax])
        if sep:
            ax.set_xlabel("group"+str(group))
        else:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

```

例如，要制作三种不同葡萄酒品种的葡萄酒样本第一个判别函数值的堆积直方图，我们键入：

```
ldahist(lda_values["x"].LD1, y)

```

![png](img/a_little_book_of_python_for_multivariate_analysis_150_0.png)

我们可以从直方图中看到，品种 1 和 3 由第一个判别函数很好地分开，因为第一个品种的值介于-6 和-1 之间，而品种 3 的值介于 2 和 6 之间，因此值之间没有重叠。

然而，线性判别函数在训练集上实现的分离可能是一个高估。为了更准确地了解第一个判别函数如何将组分开，我们需要看到三个品种的值的堆积直方图，使用一些未见的“测试集”，也就是使用未用于计算线性判别函数的数据集。

我们可以看到，第一个判别函数很好地分开了品种 1 和 3，但并没有很好地分开品种 1 和 2，或品种 2 和 3。

因此，我们调查第二个判别函数是否分开了这些品种，方法是制作第二个判别函数值的堆积直方图：

```
ldahist(lda_values["x"].LD2, y)

```

![png](img/a_little_book_of_python_for_multivariate_analysis_152_0.png)

我们看到第二个判别函数很好地区分了品种 1 和 2，尽管它们的值有一些重叠。此外，第二个判别函数也很好地区分了品种 2 和 3，尽管它们的值也有一些重叠，所以并不完美。

因此，我们看到需要两个判别函数来区分这些品种，正如上面讨论的那样（请参阅上面关于百分比分离的讨论）。### 判别函数的散点图

我们可以通过输入以下内容获得最佳两个判别函数的散点图，数据点按品种标记：

```
sns.lmplot("LD1", "LD2", lda_values["x"].join(y), hue="V1", fit_reg=False);

```

![png](img/a_little_book_of_python_for_multivariate_analysis_155_0.png)

从前两个判别函数的散点图中，我们可以看到三个品种的葡萄酒在散点图中很好地分开。第一个判别函数（x 轴）很好地区分了品种 1 和 3，但并不能完全区分品种 1 和 3，或者品种 2 和 3。

第二个判别函数（y 轴）在很大程度上很好地区分了品种 1 和 3，以及品种 2 和 3，尽管并不完全完美。

要实现三个品种的非常好的区分，最好同时使用第一个和第二个判别函数，因为第一个判别函数可以很好地区分品种 1 和 3，而第二个判别函数可以相当好地区分品种 1 和 2，以及品种 2 和 3。### 分配规则和误分类率

我们可以使用`printMeanAndSdByGroup()`函数（见上文）计算每个品种的判别函数的均值：

```
printMeanAndSdByGroup(lda_values["x"], y);

```

```
## Means:

```

|  | LD1 | LD2 |
| --- | --- | --- |
| V1 |  |  |
| --- | --- | --- |
| 1 | -3.422489 | 1.691674 |
| 2 | -0.079726 | -2.472656 |
| 3 | 4.324737 | 1.578120 |

```
## Standard deviations:

```

|  | LD1 | LD2 |
| --- | --- | --- |
| V1 |  |  |
| --- | --- | --- |
| 1 | 0.931467 | 1.008978 |
| 2 | 1.076271 | 0.990268 |
| 3 | 0.930571 | 0.971586 |

```
## Sample sizes:

```

|  | 0 |
| --- | --- |
| V1 |  |
| --- | --- |
| 1 | 59 |
| 2 | 71 |
| 3 | 48 |

我们发现第一个判别函数的均值为品种 1 为-3.42248851，品种 2 为-0.07972623，品种 3 为 4.32473717。品种 1 和 2 均值之间的中间点为（-3.42248851-0.07972623）/2=-1.751107，品种 2 和 3 均值之间的中间点为（-0.07972623+4.32473717）/2=2.122505。

因此，我们可以使用以下分配规则：

+   如果第一个判别函数小于等于-1.751107，则预测样本来自品种 1

+   如果第一个判别函数大于-1.751107 且小于等于 2.122505，则预测样本来自品种 2

+   如果第一个判别函数大于 2.122505，则预测样本来自品种 3

我们可以通过下面的`calcAllocationRuleAccuracy()`函数来检验这个分配规则的准确性：

```
def calcAllocationRuleAccuracy(ldavalue, groupvariable, cutoffpoints):
    # find out how many values the group variable can take
    levels = sorted(set((groupvariable)))
    numlevels = len(levels)
    confusion_matrix = []
    # calculate the number of true positives and false negatives for each group
    for i, leveli in enumerate(levels):
        levelidata = ldavalue[groupvariable==leveli]
        row = []
        # see how many of the samples from this group are classified in each group
        for j, levelj in enumerate(levels):
            if j == 0:
                cutoff1 = cutoffpoints[0]
                cutoff2 = "NA"
                results = (levelidata <= cutoff1).value_counts()
            elif j == numlevels-1:
                cutoff1 = cutoffpoints[numlevels-2]
                cutoff2 = "NA"
                results = (levelidata > cutoff1).value_counts()
            else:
                cutoff1 = cutoffpoints[j-1]
                cutoff2 = cutoffpoints[j]
                results = ((levelidata > cutoff1) & (levelidata <= cutoff2)).value_counts()
            try:
                trues = results[True]
            except KeyError:
                trues = 0
            print("Number of samples of group", leveli, "classified as group", levelj, ":", trues, "(cutoffs:", cutoff1, ",", cutoff2, ")")
            row.append(trues)
        confusion_matrix.append(row)
    return confusion_matrix

```

例如，要根据第一个判别函数的分配规则计算葡萄酒数据的准确性，我们输入：

```
confusion_matrix = calcAllocationRuleAccuracy(lda_values["x"].iloc[:, 0], y, [-1.751107, 2.122505])

```

```
Number of samples of group 1 classified as group 1 : 56 (cutoffs: -1.751107 , NA )
Number of samples of group 1 classified as group 2 : 3 (cutoffs: -1.751107 , 2.122505 )
Number of samples of group 1 classified as group 3 : 0 (cutoffs: 2.122505 , NA )
Number of samples of group 2 classified as group 1 : 5 (cutoffs: -1.751107 , NA )
Number of samples of group 2 classified as group 2 : 65 (cutoffs: -1.751107 , 2.122505 )
Number of samples of group 2 classified as group 3 : 1 (cutoffs: 2.122505 , NA )
Number of samples of group 3 classified as group 1 : 0 (cutoffs: -1.751107 , NA )
Number of samples of group 3 classified as group 2 : 0 (cutoffs: -1.751107 , 2.122505 )
Number of samples of group 3 classified as group 3 : 48 (cutoffs: 2.122505 , NA )

```

这可以显示在一个*混淆矩阵*中：

```
def webprint_confusion_matrix(confusion_matrix, classes_names):
    display(pd.DataFrame(confusion_matrix, index=["Is group "+i for i in classes_names], columns=["Allocated to group "+i for i in classes_names]))

webprint_confusion_matrix(confusion_matrix, lda.classes_)

```

|  | 分配给第 1 组 | 分配给第 2 组 | 分配给第 3 组 |
| --- | --- | --- | --- |
| 属于第 1 组 | 56 | 3 | 0 |
| 属于第 2 组 | 5 | 65 | 1 |
| 属于第 3 组 | 0 | 0 | 48 |

有 3+5+1=9 杯葡萄酒样本被错误分类，共有 (56+3+5+65+1+48=) 178 杯葡萄酒样本：1 杯来自品种 1 的样本被预测为来自品种 2，5 杯来自品种 2 的样本被预测为来自品种 1，1 杯来自品种 2 的样本被预测为来自品种 3。因此，误分类率为 9/178，即 5.1%。误分类率相当低，因此分配规则的准确性似乎相对较高。

然而，这可能是误分类率的低估，因为分配规则是基于这些数据（这是*训练集*）制定的。如果我们针对一个独立的*测试集*计算误分类率，该测试集包含的数据不同于用于制定分配规则的数据，我们可能会得到一个更高的误分类率估计。

#### Python 的方式

Python 允许以更快的方式进行所有上述操作，并通过使用`sklearn.metrics`模块提供扩展的自动报告功能。上述混淆矩阵和报告典型性能指标，如*精确度*、*召回率*、*F1 分数*可以在 Python 中如下完成：

```
import sklearn.metrics as metrics

def lda_classify(v, levels, cutoffpoints):
    for level, cutoff in zip(reversed(levels), reversed(cutoffpoints)):
        if v > cutoff: return level
    return levels[0]

y_pred = lda_values["x"].iloc[:, 0].apply(lda_classify, args=(lda.classes_, [-1.751107, 2.122505],)).values
y_true = y

```

```
# from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#example-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

print(metrics.classification_report(y_true, y_pred))
cm = metrics.confusion_matrix(y_true, y_pred)
webprint_confusion_matrix(cm, lda.classes_)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plot_confusion_matrix(cm_normalized, lda.classes_, title='Normalized confusion matrix')

```

```
             precision    recall  f1-score   support

          1       0.92      0.95      0.93        59
          2       0.96      0.92      0.94        71
          3       0.98      1.00      0.99        48

avg / total       0.95      0.95      0.95       178

```

|  | 分配给第 1 组 | 分配给第 2 组 | 分配给第 3 组 |
| --- | --- | --- | --- |
| 属于第 1 组 | 56 | 3 | 0 |
| 属于第 2 组 | 5 | 65 | 1 |
| 属于第 3 组 | 0 | 0 | 48 |

![png](img/a_little_book_of_python_for_multivariate_analysis_168_2.png)  ## 链接和进一步阅读

这里有一些信息和进一步阅读的链接。

要了解多元分析，我推荐以下内容：

+   [多元数据分析](http://www.bookbutler.co.uk/compare?isbn=9781292021904) [http://www.bookbutler.co.uk/compare?isbn=9781292021904]，作者是 Hair 等人。

+   [应用多元数据分析](http://www.bookbutler.co.uk/compare?isbn=9780340741221) [http://www.bookbutler.co.uk/compare?isbn=9780340741221]，作者是 Everitt 和 Dunn。

如果你是 Python 新手，你可以阅读网络上存在的大量教程之一。以下是一些链接：

+   Python 组织的[官方教程](https://docs.python.org/2/tutorial/) [https://docs.python.org/2/tutorial/]。内容广泛，几乎涵盖了核心 Python 的所有内容，并提供了许多详细的非交互式示例。

+   对于那些更喜欢通过互动教程学习的人，这里有一些我推荐的好教程：

    +   [Codecademy Python 教程](https://www.codecademy.com/learn/python) [https://www.codecademy.com/learn/python]

    +   [learnpython.org](http://www.learnpython.org)

要了解 Python 生态系统中的数据分析和数据科学，我建议您参考以下内容：

+   [《Python 数据分析》](http://shop.oreilly.com/product/0636920023784.do)由韦斯·麦金尼

+   [《从零开始的数据科学》](http://shop.oreilly.com/product/0636920033400.do)由乔尔·格鲁斯

要了解如何在 Python 中使用 scikit-learn 进行机器学习，我建议您参考：

+   [scikit-learn 首页](http://scikit-learn.org)不仅提供了优秀的文档和示例，还提供了有关机器学习方法的有用且清晰的资源。

+   [《Python 机器学习》](https://www.packtpub.com/big-data-and-business-intelligence/python-machine-learning)由塞巴斯蒂安·拉什卡

这里的可视化是使用[matplotlib](http://matplotlib.org)和[seaborn](http://stanford.edu/~mwaskom/software/seaborn/)生成的。它们的首页有关于 API 的详细文档和大量示例。

虽然 Python 是自给自足的，功能相当广泛，而且可能比任何其他科学语言都增长更快，但如果出于任何原因您需要使用 R，那么您可以通过[`rpy2`库](http://rpy2.readthedocs.org)在 Python 中使用它。## 致谢

我要感谢[艾弗里尔·科克兰](http://www.sanger.ac.uk/research/projects/parasitegenomics/)，英国剑桥的威尔康信托桑格研究所，她的优秀资源《[多元分析的 R 小书](https://little-book-of-r-for-multivariate-analysis.readthedocs.org)》，并以[CC-BY-3.0 许可证](https://creativecommons.org)发布，从而允许从 R 翻译成 Python。向她致敬。

和原版一样，本手册中的许多示例都受到 Open University 书籍《多元分析》（产品代码 M249/03）中示例的启发。

我也要感谢[UCI 机器学习库](http://archive.ics.uci.edu/ml)，提供了本手册中示例中使用的数据集。## 联系方式

如果您能给我([Yiannis Gatsoulis](http://gats.me))发送改正或改进建议至我的邮箱 gatsoulis AT gmail DOT com，我将不胜感激。## 许可证

[![知识共享许可协议](https://i.creativecommons.org/l/by-sa/4.0/88x31.png)](http://creativecommons.org/licenses/by-sa/4.0/)

由 Yiannis Gatsoulis 创作的 Python 多元分析小册子受[知识共享署名-相同方式共享 4.0 国际许可协议](http://creativecommons.org/licenses/by-sa/4.0/)许可。

基于 Avril Coghlan 的作品[A Little Book of R for Multivariate Analysis](https://little-book-of-r-for-multivariate-analysis.readthedocs.org/en/latest/src/multivariateanalysis.html)，在[CC-BY-3.0](http://creativecommons.org/licenses/by/3.0/)许可下授权。 © 版权所有 2016, Yiannis Gatsoulis。 使用[Sphinx](http://sphinx-doc.org/) 1.3.4 创建。

### 导航

+   索引

+   Python 多元分析小册子 0.1 文档 »

# 索引

© 版权所有 2016, Yiannis Gatsoulis。 使用[Sphinx](http://sphinx-doc.org/) 1.3.4 创建。
