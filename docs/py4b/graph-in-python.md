# Python 中的图形

> 原文：<https://www.pythonforbeginners.com/data-structures/graph-in-python>

图是最重要的数据结构之一。图形用于表示电话网络、地图、社交网络连接等。在本文中，我们将讨论什么是图，以及如何用 Python 实现图。

## 什么是图？

在数学中，图被定义为一组顶点和边，其中顶点是特定的对象，边表示顶点之间的连接。顶点和边用集合来表示。

数学上，一个图 G 可以表示为 G= (V，E)，其中 V 是顶点的集合，E 是边的集合。

如果一条边 E[i] 连接顶点 v1 和 v2，我们可以把这条边表示为 E[i] = (v1，v2)。

## 如何表示一个图形？

我们将使用下图中给出的图形来学习如何表示图形。

![Graph in Python](img/508fe290a193f247bba06c8c13736bfa.png)



Graph in Python

为了表示一个图，我们必须找到图中的顶点和边的集合。

首先，我们将找到顶点集。为此，我们可以使用上图中给出的顶点创建一个集合。在图中，顶点被命名为 A、B、C、D、E 和 F，因此顶点集可以创建为 V={A，B，C，D，E，F}。

为了找到边的集合，首先我们将找到图中的所有边。你可以观察到图中有 6 条边，编号从 E ₁ 到 E₆ 。边 E[i] 可以被创建为元组(v1，v2 ),其中 v1 和 v2 是由 E[i] 连接的顶点。对于上图，我们可以将边表示如下。

*   E₁ = (A，D)
*   E₂ = (A，B)
*   E₃ = (A，E)
*   E₄ = (A，F)
*   E₅ = (B，F)
*   E₆ = (B，C)

边 E 的集合可以表示为 E= {E ₁ ，E ₂ ，E₃ ，E₄ ，E₅ ，E₆ }。

最后，图 G 可以表示为 G= (V，E)其中 V 和 E 是顶点和边的集合。

到目前为止，我们已经讨论了如何用数学方法表示一个图。你能想出一种在 python 程序中表示图形的方法吗？让我们来研究一下。

## 如何用 Python 表示一个图？

我们可以用邻接表来表示一个图。邻接表可以被认为是一个列表，其中每个顶点都存储了与之相连的所有顶点的列表。

我们将使用字典和列表在 python 中实现图的邻接表表示。

首先，我们将使用给定的顶点集创建一个 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)，其中所有顶点名称作为关键字，一个空列表(邻接表)作为它们的关联值。

之后，我们将使用给定的边集来完成每个顶点的邻接表，该邻接表已经使用字典的关键字来表示。对于每条边(v1，v2)，我们会将 v1 添加到 v2 的邻接表中，将 v2 添加到 v1 的邻接表中。

这样，字典中的每个键(顶点)都将有一个关联的值(一个顶点列表),字典将用 python 表示整个图。

给定顶点和边的集合，我们可以如下用 python 实现一个图。

```py
vertices = {"A", "B", "C", "D", "E", "F"}
edges = {("A", "D"), ("A", "B"), ("A", "E"), ("A", "F"), ("B", "F"), ("B", "C")}
graph = dict()
for vertex in vertices:
    graph[vertex] = []
for edge in edges:
    v1 = edge[0]
    v2 = edge[1]
    graph[v1].append(v2)
    graph[v2].append(v1)
print("The given set of vertices is:", vertices)
print("The given set of edges is:", edges)
print("Graph representation in python is:")
print(graph) 
```

输出:

```py
The given set of vertices is: {'F', 'D', 'B', 'E', 'A', 'C'}
The given set of edges is: {('A', 'F'), ('A', 'B'), ('B', 'C'), ('A', 'D'), ('A', 'E'), ('B', 'F')}
Graph representation in python is:
{'F': ['A', 'B'], 'D': ['A'], 'B': ['A', 'C', 'F'], 'E': ['A'], 'A': ['F', 'B', 'D', 'E'], 'C': ['B']}
```

在上面的输出中，您可以验证图形的每个关键点都有一个连接到它的顶点列表作为它的值。

## 结论

在本文中，我们讨论了图形数据结构。我们还讨论了图形的数学表示，以及如何用 python 实现它。要了解更多关于 Python 中数据结构的知识，可以阅读这篇关于 python 中[链表的文章。](https://www.pythonforbeginners.com/lists/linked-list-in-python)