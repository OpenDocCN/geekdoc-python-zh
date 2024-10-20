# 用 Python 实现图形

> 原文：<https://www.askpython.com/python/examples/graph-in-python>

图形是一种数据结构，用于说明两个对象之间的联系。图的一个简单例子是地理地图，其中不同的地方用道路连接起来。在本文中，我们将研究图数据结构的理论方面。此外，我们将使用两种不同的方法实现一个图形。

## 什么是图？

图形是一种非线性数据结构，用于表示相互连接的对象。对象被称为顶点，它们之间的链接被称为边。

在数学上，图 G 被定义为两个集合 V 和 E 的有序对。它被表示为 G=(V，E)其中，

*   v 是图中顶点的集合。
*   e 是图中存在的边的集合。每条边都用一个元组来表示，元组显示了它所连接的顶点。例如，如果边“e”连接顶点 v1 和 v2，它将被表示为(v1，v2)。

为了更清楚地理解这一点，让我们看下面的例子。

![Graph Implementation In Python- Askpython](img/5881ab97e34a7a44225b1fb5ded95f10.png)

Graph Implementation In Python – Askpython

在上图中，我们有一个包含 6 个顶点的图，即 0，1，2，3，4，5。因此，等式 G=(V，E)中的集合 V 将是顶点的集合，其将被表示如下。

```py
V={0,1,2,3,4,5}

```

为了找到由边组成的集合 E，我们将首先找到每条边。在上图中，我们有 8 条线连接图形的不同顶点。

我们使用顶点连接的名称来定义每个顶点“v”。例如，连接 0 到 1 的边将被称为 e01，并且将使用元组(0，1)来表示。类似地，所有的边将被定义如下。

```py
e01=(0,1)
e12=(1,2)
e03=(0,3)
e13=(1,3)
e34=(3,4)
e25=(2,5)
e45=(4,5)
e24=(2,4)

```

由图中每条边组成的集合 E 定义如下。

```py
 E={(0,1),(1,2),(0,3),(1,3),(3,4),(2,5),(4,5),(2,4)}.

```

由于我们已经获得了该图的数学符号，现在我们将用 python 实现它。

## 如何用 Python 实现一个使用邻接矩阵的图？

如果我们有一个有 N 个顶点的图，这个图的邻接矩阵将是一个二维矩阵。矩阵中的行和列表示图的顶点，矩阵中的值决定两个顶点之间是否有边。

假设我们有任何图的邻接矩阵 A。对于任意索引(I，j)，如果顶点 I 和顶点 j 之间有边，我们给 A[i][j]赋值 1。当顶点 I 和 j 之间不存在边时，值 0 被分配给 A[i][j]。这可以用 Python 实现，如下所示。

```py
import numpy as np

# keep vertices in a set
vertices = {0, 1, 2, 3, 4, 5}
# keep edges in a set
edges = {(0, 1), (1, 2), (0, 3), (1, 3), (3, 4), (2, 5), (4, 5), (2, 4)}
# create a 6X6 integer numpy array with all values initialised to zero
adjacencyMatrix = np.zeros((6, 6)).astype(int)
# Represent edges in the adjacency matrix
for edge in edges:
    v1 = edge[0]
    v2 = edge[1]
    adjacencyMatrix[v1][v2] = 1
    adjacencyMatrix[v2][v1] = 1 # if v1 is connected to v2, v2 is also connected to v1
print("The set of vertices of the graph is:")
print(vertices)
print("The set of edges of the graph is:")
print(edges)
print("The adjacency matrix representing the graph is:")
print(adjacencyMatrix)

```

输出:

```py
The set of vertices of the graph is:
{0, 1, 2, 3, 4, 5}
The set of edges of the graph is:
{(0, 1), (2, 4), (1, 2), (3, 4), (0, 3), (4, 5), (2, 5), (1, 3)}
The adjacency matrix representing the graph is:
[[0 1 0 1 0 0]
 [1 0 1 1 0 0]
 [0 1 0 0 1 1]
 [1 1 0 0 1 0]
 [0 0 1 1 0 1]
 [0 0 1 0 1 0]]

```

使用邻接矩阵实现图有一个缺点。这里，我们为每个顶点分配内存，不管它是否存在。这可以通过使用邻接表实现图来避免，如下节所述。

## 如何用 Python 实现一个使用邻接表的图？

邻接表存储每个顶点的所有连接顶点的列表。为了实现这一点，我们将使用一个字典，其中字典的每个键代表一个顶点，键的值包含该关键顶点所连接的顶点的列表。这可以如下实现。

```py
# keep vertices in a set
vertices = {0, 1, 2, 3, 4, 5}
# keep edges in a set
edges = {(0, 1), (1, 2), (0, 3), (1, 3), (3, 4), (2, 5), (4, 5), (2, 4)}
# create a dictionary with vertices of graph as keys and empty lists as values
adjacencyList={}
for vertex in vertices:
    adjacencyList[vertex]=[]
# Represent edges in the adjacency List
for edge in edges:
    v1 = edge[0]
    v2 = edge[1]
    adjacencyList[v1].append(v2)
    adjacencyList[v2].append(v1) # if v1 is connected to v2, v2 is also connected to v1
print("The set of vertices of the graph is:")
print(vertices)
print("The set of edges of the graph is:")
print(edges)
print("The adjacency List representing the graph is:")
print(adjacencyList)

```

输出:

```py
The set of vertices of the graph is:
{0, 1, 2, 3, 4, 5}
The set of edges of the graph is:
{(0, 1), (2, 4), (1, 2), (3, 4), (0, 3), (4, 5), (2, 5), (1, 3)}
The adjacency List representing the graph is:
{0: [1, 3], 1: [0, 2, 3], 2: [4, 1, 5], 3: [4, 0, 1], 4: [2, 3, 5], 5: [4, 2]}

```

在上面的输出中，我们可以看到每个关键点代表一个顶点，并且每个关键点都与它所连接的顶点列表相关联。这种实现比图邻接矩阵表示更有效。这是因为我们不需要存储不存在的边的值。

## 结论

在本文中，我们研究了表示图的理论概念，然后用 python 实现了一个使用邻接矩阵和邻接表表示的图。请继续关注更多内容丰富的文章。快乐学习。