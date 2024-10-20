# 如何在 Python 中初始化 2D 列表？

> 原文：<https://www.pythoncentral.io/how-to-initialize-a-2d-list-in-python/>

[![python 2d](img/d65a56486003377896789f8453b7e80b.png)](https://www.pythoncentral.io/wp-content/uploads/2021/09/pexels-christina-morillo-1181675.jpg)

列表是一种用于线性存储多个值的数据结构。然而，存在二维数据。需要一个多维数据[结构](https://www.pythoncentral.io/priority-queue-beginners-guide/)来保存这类数据。

在 Python 中，二维列表是一种重要的数据结构。要有效地使用 Python 2D 列表，学习 Python 1D 列表的工作原理是非常重要的。我们称 Python 2D 列表为嵌套列表和列表列表。

在下面的文章中，我们将学习如何使用 Python 2D 列表。会有更好学习的例子。

可以访问 [Codeleaks.io](https://www.codeleaks.io/) 获取 Python 详细教程及示例。

## **Python 1D 榜 vs 2D 榜**

Python 列表的一维列表如下所示:

List= [ 2，4，8，6 ]

另一方面，Python 2D 列表如下所示:

List= [ [ 2，4，6，8 ]，[ 12，14，16，18 ]，[ 20，40，60，80 ] ]

## **如何初始化 Python 2D 列表？**

Python 提供了多种在 Python 中初始化 2D 列表的技术。列表理解用于返回一个列表。Python 2D 列表由嵌套列表作为其元素组成。

让我们逐一讨论每种技术。

### **01 号技术**

该技术使用列表理解来创建 Python 2D 列表。这里我们使用嵌套列表理解来初始化一个二维列表。

行数= 3

列= 4

Python_2D_list = [[范围(列)中的 j 为 5]

对于范围内的 I(行)]

打印(Python _ 2D _ 列表)

**输出:**

[[5, 5, 5, 5], [5, 5, 5, 5], [5, 5, 5, 5]]

### **02 号技术**

行数= 2

列= 3

python _ 2D _ list =[[7]*列]*行

打印(Python _ 2D _ 列表)

**输出:**

[[7, 7, 7], [7, 7, 7]]

### **03 号技术**

行数= 6

列= 6

Python_2D=[]

对于范围内的 I(行):

列= []

对于范围内的 j(列):

column.append(3)

Python_2D.append(列)

打印(Python_2D)

**输出:**

[[3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3]]

### **Python 2D 列表的应用**

现在我们将列出 Python 2D 列表的应用。

1.  游戏板
2.  表列数据
3.  数学中的矩阵
4.  网格
5.  web 开发中的 DOM 元素
6.  科学实验数据

还有很多。

### **结论**

Python 2D 列表有其局限性和优点。Python 2D 列表的使用取决于 Python 程序的要求。我希望这篇文章能帮助你学习 Python 2D 列表的概念。