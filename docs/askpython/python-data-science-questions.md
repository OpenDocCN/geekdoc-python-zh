# Python 数据科学问题

> 原文：<https://www.askpython.com/python/python-data-science-questions>

在这一节中，我们将讨论什么是 python，它的历史、起源、当前版本、薪水和 2022 年的工作角色，然后我们将进入重要的 Python 编程问题。

近年来，Python 已经成为世界上最流行的编程语言之一。它在全球众多设备上使用。由于可用库的范围很广，开发人员和非开发人员都可以使用它。

Python 是一种计算机编程语言，通常用于构建网站和软件、自动化任务以及行为记录分析。它是一种通用语言，这意味着它可以用来创建各种不同的程序，而不是专门针对任何特定的问题。这种多功能性，加上它对初学者的友好性，使它成为当今最常用的编程语言之一。在世界各地不同组织提供的许多调查中，Python 成为 2022 年最受欢迎的语言。

Python 是由**吉多·范·罗苏姆**于 20 世纪 80 年代末在*荷兰国家数学和计算研究院*开发的。它继承了 **ABC** 编程语言，该语言与阿米巴操作系统接口，并具有出色的处理能力。

Python 3.10.7 是 Python 编程语言的最新版本，包括许多新特性和优化。

## 2022 年薪酬最高的 Python 职位

*   人工智能专家|＄135，238
*   解决方案架构师|＄120，756
*   机器学习工程师|＄112343
*   分析经理|＄99，121
*   数据科学家|＄97004
*   数据工程师|＄92，999
*   软件工程师|＄88280
*   后端开发人员|＄87，009
*   计算机科学家|＄81812
*   前端开发人员|＄76，289

## 理论 Python 数据科学问题

### 1.我们使用哪个库进行数据操作？

[Pandas 是 Python 的库](https://www.askpython.com/python-modules/pandas/python-pandas-module-tutorial)。pandas 是一个非常受欢迎的库，它与 NumPy 和 [matplotlib](https://www.askpython.com/python-modules/matplotlib/python-matplotlib) 一起被广泛用于数据科学。它有一个活跃的社区，有 1000 多名贡献者，主要用于数据分析和清理。

### 2.用 Python 写数据科学的前 5 个库。

在数据科学项目中广泛使用的 5 大 Python 库是:

*   TensorFlow
*   熊猫
*   NumPy
*   Matplotlib
*   [轨道](https://www.askpython.com/python-modules/python-scipy)轨道

### 3.数列和向量有什么区别？

*   **向量**只分配索引位置值为 0，1，…，(n-1)。
*   **系列**只有一列。它为每个数据系列分配自定义索引位置值。例如:客户标识、客户名称、总销售额。系列可以从列表、数组、字典中创建。

### 4.区分数据框和矩阵。

**数据帧**

*   数据框是共享一个公共索引的系列的集合。
*   它可以保存多个不同数据类型的序列。
*   例如，雇员数据有各种列，如雇员 id、雇员姓名、年龄、性别和部门。这些分别是不同数据类型的系列。

**矩阵**

*   Numpy 中的矩阵由多个向量构成。
*   在整个二维结构中，它只能容纳一种数据类型。

### 5.解释熊猫数据帧分组的用途。

Groupby 允许根据列将行组合在一起，并对这些组合行执行聚合函数。示例:df.groupby('salary ')。平均值()。

### 6.说出一些可以用于可视化的 Python 库。

[**Matplotlib**](https://www.askpython.com/python-modules/matplotlib/python-matplotlib) 是一个标准的数据可视化库，对于生成二维图形非常有用。例如:直方图、饼图、条形图、柱形图和散点图。很多库都是在 Matplotlib 之上构建的，它的函数可以在后端使用。此外，它还广泛用于创建可视化的轴和布局。

[**Seaborn**](https://www.askpython.com/python-modules/python-seaborn-tutorial) 基于 Matplotlib。它是 Python 中的一个数据可视化库。它适用于 Numpy 和 Pandas，并且为绘制有吸引力的和信息丰富的统计图形提供了一个很好的界面。

### 7.什么是散点图？

二维数据可视化解释了两个不同变量的观察值之间的关系。一个绘制在 x 轴上，另一个绘制在 y 轴上。

### 8.regplot()、lmplot()和 residplot()有什么区别？

*   **regplot()** 是用于绘制数据的**和拟合**的线性回归模型。为了估计回归模型，有几种互斥的可能性。
*   **lmplot()** 绘制数据，回归模型拟合一个面网格**。**它被设计为一个实用的接口，用于跨数据集的条件子集拟合回归模型，并且计算量更大。lmplot()结合了 regplot()和 FacetGrid。
*   **residplot()** 绘制 X 和 Y 之间的误差，为其创建一个线性回归方程。

### 9.定义热图。

热图是一种数据可视化，它利用颜色来描述一个值如何根据另外两个变量的值而变化。例如，您可以使用热图来了解一组城市的气温如何随时间变化。

### 10.为什么使用 Python 而不是其他语言？

Python 是一种广泛使用、灵活且通用的编程语言。因为它简单明了，所以作为第一语言是很棒的。它也是任何程序员工具箱中的有用语言，因为它可以用于从 web 开发到软件开发到科学应用的任何事情。

### 11.Python 中的枚举函数是什么？

Python enumerate()为 iterable 增加一个计数器，并以枚举对象的形式返回。枚举对象可以直接用于循环，或者使用`list()`方法转换成元组列表

### 12.复数绝对值背后的数学原理是什么？

如果`**z=a+ib**`，则绝对值计算为 **`sqrt(a^2+b^2)`**

### 13.Python 中有哪些顶级的文本挖掘库？

*   自然语言工具包(NLTK)
*   Gensim
*   -好的
*   宽大的
*   文本 Blob
*   模式
*   PyNLPl

### 14.熊猫在数据分析中是如何使用的？

Pandas 使得使用类似 SQL 的查询来加载、处理和分析这样的表格数据变得非常方便。Pandas 为表格数据的可视化分析提供了多种选项，可与 Matplotlib 和 Seaborn 配合使用。Pandas 中的主要数据结构是用 Series 和 DataFrame 类实现的。

### 15.说出五大 Python 编译器。

*   皮查姆
*   崇高的文本
*   托尼
*   Visual Studio 代码
*   Jupyter 笔记型电脑

### 16.Python 中的关键词是什么？

Python 使用具有特定含义的保留字，称为关键字。它们通常用于指定变量的种类。变量名和函数名不允许包含关键字。下面列出的 33 个关键词都是 Python 中的:

| 或者 | 和 | 不 | 如果 | 艾列弗 |
| 其他 | 为 | 在…期间 | 破裂 | 极好的 |
| 如同 | 希腊字母的第 11 个 | 及格 | 返回 | 真实的 |
| 错误的 | 随着 | 尝试 | 维护 | 班级 |
| 继续 | 是吗 | 除...之外 | 最后 | 从 |
| 全球的 | 进口 | 在 | 是 | 没有人 |
| 非局部的 | 上升 | 产量 |  |  |

Python Keywords

## 数据科学:编码问题

### 1.用 Python 写一个程序预测输出类型

```py
# Defining the variable  
x = 'z'
print(type(x))

```

### 2.编写一个 python 程序，使用 while 循环打印一个 13 的表。

```py
i = 0
while i <= 10:
    print(i*13)
    i+=1

```

### 3.如何用 Python 访问 CSV 文件？

*   通过使用 [CSV 库](https://www.askpython.com/python-modules/pandas/pandas-read-csv-with-headers)

```py
import csv

with open("bwq.csv", 'r') as file:
  csv_reader = csv.reader(file)
  for row in csvreader:
    print(row)

```

*   使用[熊猫图书馆](https://www.askpython.com/python-modules/pandas/python-pandas-module-tutorial)

```py
import pandas as pd
data_bwq = pd.read_csv("bwq.csv")
data_bwq

```

### 4.用 Python 生成随机数。

```py
#generating random numbers between (0,22)
import random
n = random.randint(0,22)
print(n)

```

### 5.检查元素是否在序列中。

```py
42 in [2, 39, 42]

# Output: True

```

### 6.展示 extend 和 append 函数的区别。

*   append:它将对象附加在末尾。

```py
a = [1, 2, 3]
a.append([4, 5])
print (a)

# Output: [1, 2, 3, [4, 5]]

```

*   extend:通过追加 iterable 中的元素来扩展列表。

```py
a = [1, 2, 3]
a.extend([4, 5])
print (a)

# Output: [1, 2, 3, 4, 5]

```

### 7.打印 10 到 100 的所有倍数。

```py
multiples=[] 
for i in range(10, 101): 
    if i%10==0: 
        multiples.append(i) 
print(multiples)

# Output: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

```

### 8.修复 Python 中的 ModuleNotFoundError 和 ImportError。

*   首先，确保使用绝对导入
*   其次，将项目的根目录导出到`PYTHONPATH`

大多数现代 Python IDEs 会自动完成这一任务，但如果不是这样，我确信会有这样一个选项，您可以在其中为您的 Python 应用程序定义`PYTHONPATH`(至少是 PyCharm)。如果您在另一个环境中运行 Python 应用程序，比如 Docker、vagger，或者在您的虚拟环境中，您可以在 bash 中运行以下命令:

```py
export PYTHONPATH="${PYTHONPATH}:/path/to/your/project/"
# * For Windows
set PYTHONPATH=%PYTHONPATH%;C:\path\to\your\project\

```

### 9.编写方法来分隔具有特定扩展名(.csv，。txt)存储在一个目录中

*   **方法 1**

```py
import os
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(‘.txt’):
            print file

```

*   **方法二**

```py
import os
path = ‘mypath/path’
files = os.listdir(path)
files_txt = [i for i in files if i.endswith(‘.txt’)

```

## 结论

以上是数据科学面试中一些最常被问到的问题。还有许多其他的例子，但 Python 的基础知识是面对数据科学面试的基本要求。文档参考也是获得数据科学领域中使用的多个库的更好工作知识所需的关键技能之一。