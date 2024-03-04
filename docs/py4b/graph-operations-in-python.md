# Python 中的图形操作

> 原文：<https://www.pythonforbeginners.com/data-structures/graph-operations-in-python>

图形是一种非线性数据结构，用于表示不同对象之间的联系。通常，图表用于表示地图、网络和社交媒体连接。在本文中，我们将研究如何在 Python 中执行不同的图形操作。我们将获取一个图，并将其作为一个运行示例来执行所有的图操作。

## 什么是不同的图形操作？

图通常以邻接表的形式提供。如果我们讨论上的运算，可能有如下的图形运算。

*   打印图形的所有顶点
*   打印图表的所有边
*   在图中插入一个顶点
*   在图中插入一条边

我们将在 python 中执行所有这些图形操作。为此，我们将使用下图中给出的图表。

![Graph in Python](img/8f9872874a2835a02d2dad9cd058d424.png)



Graph in Python

在执行图操作之前，我们将如下构建上述图的邻接表表示。

```py
graph = {'A': ['B', 'D', 'E', 'F'], 'D': ['A'], 'B': ['A', 'F', 'C'], 'F': ['B', 'A'], 'C': ['B'], 'E': ['A']}
print("Graph representation in python is:")
print(graph) 
```

输出:

```py
Graph representation in python is:
{'A': ['B', 'D', 'E', 'F'], 'D': ['A'], 'B': ['A', 'F', 'C'], 'F': ['B', 'A'], 'C': ['B'], 'E': ['A']} 
```

## 如何打印一个图的所有顶点

从上一篇关于 python 中的图的文章中，我们知道图的顶点是使用邻接矩阵(这是一个 python 字典)的键来表示的。因此，我们可以通过简单地打印邻接矩阵的键来打印图的所有顶点。为此，我们将如下使用 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)的 keys()方法。

```py
graph = {'A': ['B', 'D', 'E', 'F'], 'D': ['A'], 'B': ['A', 'F', 'C'], 'F': ['B', 'A'], 'C': ['B'], 'E': ['A']}
print("Graph representation in python is:")
print(graph)
vertices= list(graph.keys())
print("Vertices in the graph are:",vertices) 
```

输出:

```py
Graph representation in python is:
{'A': ['B', 'D', 'E', 'F'], 'D': ['A'], 'B': ['A', 'F', 'C'], 'F': ['B', 'A'], 'C': ['B'], 'E': ['A']}
Vertices in the graph are: ['A', 'D', 'B', 'F', 'C', 'E']
```

## 如何打印图形的所有边

我们知道，图中的边是通过使用与每个顶点相关联的列表来表示的。每个顶点都存储一个与之相连的顶点列表。我们将遍历每个顶点 v1，并为存在于与 v1 相关联的列表中的每个顶点创建边(v1，v2)。

记住，当打印边时，会有重复，因为每当顶点 v2 出现在与 v1 相关联的列表中时，v1 也会出现在与 v2 相关联的列表中。因此，在打印边缘时，将打印(v1，v2)和(v2，v1)两者，这引入了冗余，因为(v1，v2)和(v2，v1)两者表示相同的边缘。

为了克服这个问题，我们将把任何边(v1，v2)存储为一个无序集。这样，(v1，v2)将与(v2，v1)相同。之后，我们将创建一个边列表。在向列表中插入任何新边之前，我们将首先检查该边是否存在于列表中。如果列表中已经存在任何边，我们将不会插入任何重复的边。

上述过程可以用 python 实现如下。

```py
graph = {'A': ['B', 'D', 'E', 'F'], 'D': ['A'], 'B': ['A', 'F', 'C'], 'F': ['B', 'A'], 'C': ['B'], 'E': ['A']}
print("Graph representation in python is:")
print(graph)
edges = []
for vertex in graph:
    for adjacent_vertex in graph[vertex]:
        edge = {vertex, adjacent_vertex}
        if edge not in edges:
            edges.append(edge)

print("Edges in the graph are:", edges) 
```

输出:

```py
Graph representation in python is:
{'A': ['B', 'D', 'E', 'F'], 'D': ['A'], 'B': ['A', 'F', 'C'], 'F': ['B', 'A'], 'C': ['B'], 'E': ['A']}
Edges in the graph are: [{'B', 'A'}, {'A', 'D'}, {'A', 'E'}, {'A', 'F'}, {'B', 'F'}, {'B', 'C'}]
```

## 如何在图中插入一个顶点

我们知道顶点是用邻接表的关键字来表示的。为了将一个顶点插入到图中，我们将把这个顶点作为一个键插入到图中，用一个空列表作为它的关联值。空列表表示当前顶点没有连接到任何其他顶点。我们可以这样实现它。

```py
graph = {'A': ['B', 'D', 'E', 'F'], 'D': ['A'], 'B': ['A', 'F', 'C'], 'F': ['B', 'A'], 'C': ['B'], 'E': ['A']}
print("Original Graph representation is:")
print(graph)
# insert vertex G
graph["G"] = []
print("The new graph after inserting vertex G is:")
print(graph) 
```

输出:

```py
Original Graph representation is:
{'A': ['B', 'D', 'E', 'F'], 'D': ['A'], 'B': ['A', 'F', 'C'], 'F': ['B', 'A'], 'C': ['B'], 'E': ['A']}
The new graph after inserting vertex G is:
{'A': ['B', 'D', 'E', 'F'], 'D': ['A'], 'B': ['A', 'F', 'C'], 'F': ['B', 'A'], 'C': ['B'], 'E': ['A'], 'G': []} 
```

## 如何在图形中插入一条边

在图中插入边比打印边要简单得多。我们知道每个顶点包含一个它所连接的顶点列表。因此，为了插入边(v1，v2)，我们将简单地将顶点 v1 插入到与 v2 相关联的顶点列表中，并将 v2 插入到与顶点 v1 相关联的顶点列表中。

这样，将建立 v1 连接到 v2，v2 连接到 v1。因此，顶点(v1，v2)将被添加到图形中。我们可以这样实现它。

```py
graph ={'A': ['B', 'D', 'E', 'F'], 'D': ['A'], 'B': ['A', 'F', 'C'], 'F': ['B', 'A'], 'C': ['B'], 'E': ['A'], 'G': []}
print("Original Graph representation is:")
print(graph)
# insert vertex (D,G)
graph["D"].append("G")
graph["G"].append("D")
print("The new graph after inserting edge (D,G) is:")
print(graph) 
```

输出:

```py
Original Graph representation is:
{'A': ['B', 'D', 'E', 'F'], 'D': ['A'], 'B': ['A', 'F', 'C'], 'F': ['B', 'A'], 'C': ['B'], 'E': ['A'], 'G': []}
The new graph after inserting edge (D,G) is:
{'A': ['B', 'D', 'E', 'F'], 'D': ['A', 'G'], 'B': ['A', 'F', 'C'], 'F': ['B', 'A'], 'C': ['B'], 'E': ['A'], 'G': ['D']}
```

## 结论

在本文中，我们用 python 实现了不同的图形操作。想了解更多关于其他数据结构的知识，可以阅读这篇关于 python 中[链表的文章。](https://www.pythonforbeginners.com/lists/linked-list-in-python)