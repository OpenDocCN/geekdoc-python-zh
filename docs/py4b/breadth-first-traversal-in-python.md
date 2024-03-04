# Python 中的广度优先遍历

> 原文：<https://www.pythonforbeginners.com/data-structures/breadth-first-traversal-in-python>

图形是一种非线性数据结构。我们经常使用图形来表示不同的现实世界对象，如地图和网络。在本文中，我们将研究广度优先遍历来打印一个图中的所有顶点。我们还将在 Python 中实现广度优先遍历算法。

## 什么是广度优先遍历？

广度优先遍历是一种打印图中所有顶点的图遍历算法。在这个算法中，我们从一个顶点开始并打印它的值。然后我们打印当前顶点的所有邻居。之后，我们选择当前顶点的每个邻居，并打印其所有邻居。这个过程一直持续到图形中的所有顶点都被打印出来。

让我们通过在下图中执行广度优先遍历来理解这个过程。

![](img/034cf5399a74b084b1f802b0539cdab1.png)



Image of a Graph

假设我们从顶点 a 开始。

打印完顶点 A 后，我们将打印它的所有相邻顶点，即 B、D、E 和 f。

在打印 B、D、E 和 F 之后，我们将选择这些顶点中的一个。让我们选择 d。

由于 D 没有需要打印的邻居，我们将返回到 A 并选择 A 的另一个邻居。

由于 E 没有需要打印的邻居，我们将回到 A 并选择 A 的另一个邻居。

由于 F 没有需要打印的邻居，我们将回到 A 并选择 A 的另一个邻居。

现在，我们将打印 B 的所有尚未打印的邻居。因此，我们将打印 c。

此时，您可以观察到图中的所有顶点都已按 A、B、D、E、F、C 的顺序打印出来。因此，我们将终止该算法。

## 广度优先遍历算法

使用[队列数据结构](https://www.pythonforbeginners.com/queue/queue-in-python)实现图的深度优先遍历算法。这里，我们假设我们有一个连通图。换句话说，我们可以从起始顶点到达图的每个顶点。

我们将维护一个队列来存储尚未打印的顶点，并维护一个列表来存储已访问的顶点。之后，我们将使用以下算法处理该图。

1.  创建一个空队列 Q 来存储没有被打印的顶点。
2.  创建一个空列表 L 来存储访问过的顶点。
3.  将源顶点插入 Q 和 l。
4.  如果 Q 为空，请转到 9。否则转到 5。
5.  从 q 中取出一个顶点 v。
6.  打印顶点 v。
7.  将 v 的所有不在 L 中的邻居也插入到 Q 和 L 中。
8.  转到第 4 节。
9.  停下来。

可以使用上图中给出的图形的源代码来演示该算法。

```py
from queue import Queue

graph = {'A': ['B', 'D', 'E', 'F'], 'D': ['A'], 'B': ['A', 'F', 'C'], 'F': ['B', 'A'], 'C': ['B'], 'E': ['A']}
print("Given Graph is:")
print(graph)

def BFS_Algorithm(input_graph, source):
    Q = Queue()
    visited_vertices = list()
    Q.put(source)
    visited_vertices.append(source)
    while not Q.empty():
        vertex = Q.get()
        print("At:",vertex)
        print("Printing vertex:",vertex)
        for u in input_graph[vertex]:
            if u not in visited_vertices:
                print("At vertex, adding {} to Q and visited_vertices".format(vertex, u))
                Q.put(u)
                visited_vertices.append(u)
        print("visited vertices are: ", visited_vertices)

print("BFS traversal of graph with source A is:")
BFS_Algorithm(graph, "A") 
```

输出:

```py
Given Graph is:
{'A': ['B', 'D', 'E', 'F'], 'D': ['A'], 'B': ['A', 'F', 'C'], 'F': ['B', 'A'], 'C': ['B'], 'E': ['A']}
BFS traversal of graph with source A is:
At: A
Printing vertex: A
At vertex, adding A to Q and visited_vertices
At vertex, adding A to Q and visited_vertices
At vertex, adding A to Q and visited_vertices
At vertex, adding A to Q and visited_vertices
visited vertices are:  ['A', 'B', 'D', 'E', 'F']
At: B
Printing vertex: B
At vertex, adding B to Q and visited_vertices
visited vertices are:  ['A', 'B', 'D', 'E', 'F', 'C']
At: D
Printing vertex: D
visited vertices are:  ['A', 'B', 'D', 'E', 'F', 'C']
At: E
Printing vertex: E
visited vertices are:  ['A', 'B', 'D', 'E', 'F', 'C']
At: F
Printing vertex: F
visited vertices are:  ['A', 'B', 'D', 'E', 'F', 'C']
At: C
Printing vertex: C
visited vertices are:  ['A', 'B', 'D', 'E', 'F', 'C'] 
```

在输出中，您可以观察到顶点是按照 A、B、D、E、F 和 c 的顺序打印的。

## 广度优先遍历在 Python 中的实现

我们已经讨论了图的广度优先遍历的一般思想，并观察了该算法如何使用 python 程序工作，我们可以如下实现广度优先遍历算法。

```py
from queue import Queue

graph = {'A': ['B', 'D', 'E', 'F'], 'D': ['A'], 'B': ['A', 'F', 'C'], 'F': ['B', 'A'], 'C': ['B'], 'E': ['A']}
print("Given Graph is:")
print(graph)

def BFS(input_graph, source):
    Q = Queue()
    visited_vertices = list()
    Q.put(source)
    visited_vertices.append(source)
    while not Q.empty():
        vertex = Q.get()
        print(vertex, end= " ")
        for u in input_graph[vertex]:
            if u not in visited_vertices:
                Q.put(u)
                visited_vertices.append(u)

print("BFS traversal of graph with source A is:")
BFS(graph, "A") 
```

输出:

```py
Given Graph is:
{'A': ['B', 'D', 'E', 'F'], 'D': ['A'], 'B': ['A', 'F', 'C'], 'F': ['B', 'A'], 'C': ['B'], 'E': ['A']}
BFS traversal of graph with source A is:
A B D E F C 
```

## 结论

在本文中，我们讨论了 python 中全连通图的广度优先遍历算法。要了解更多关于其他算法的内容，可以阅读这篇关于 Python 中的[有序树遍历](https://www.pythonforbeginners.com/data-structures/in-order-tree-traversal-in-python)的文章。