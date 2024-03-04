# 图中从一个顶点到其它顶点的最短路径长度

> 原文：<https://www.pythonforbeginners.com/data-structures/shortest-path-length-from-a-vertex-to-other-vertices-in-a-graph>

图形用于表示地理地图、计算机网络等。在本文中，我们将讨论如何计算未加权图中顶点之间的最短距离。为了计算从一个顶点到其他顶点的最短路径长度，我们将使用广度优先搜索算法。

## 如何计算一个顶点到其他顶点的最短路径长度？

在未加权的图中，所有的边都有相等的权重。这意味着我们只需计算每个顶点之间的边数，就可以计算出它们之间的最短路径长度。

例如，考虑下图。

![Graph in Python](img/cc4280cbd1055847202dbea43ac16b64.png)



Graph in Python

让我们计算上图中每个顶点之间的最短距离。

顶点 A 和顶点 b 之间只有一条边 E ₂ 所以，它们之间的最短路径长度是 1。

我们可以用两种方法从 A 到达 C。第一种是使用边 E₄->E₅->E₆，第二种是使用边 E ₂ - > E₆ 。这里我们将选择最短路径，即 E ₂ - > E₆ 。因此，顶点 A 和顶点 C 之间的最短路径长度是 2。

顶点 A 和顶点 d 之间只有一条边 E ₁ 所以，它们之间的最短路径长度是 1。

顶点 A 和顶点 E 之间只有一条边 E₃ 所以，它们之间的最短路径长度是 1。

我们可以用两种方法从 A 到达 F。第一条路径使用边 E ₂ - > E₅ ，第二条路径使用边 E₄ 。这里，我们将选择最短路径，即₄ 。因此，顶点 A 和顶点 F 之间的最短路径长度是 1。

## 计算从一个顶点到其它顶点最短路径长度的算法

到现在为止，你一定已经明白我们必须计算顶点之间的边数来计算顶点之间的距离。为此，我们将如下修改广度优先搜索算法。

*   我们将声明一个 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)，它将包含顶点作为它们的键，以及到源顶点的距离作为关联值。
*   最初，我们将把每个顶点到源的距离指定为无穷大，用一个大的数字表示。每当我们在遍历过程中找到一个顶点时，我们将计算该顶点到源的当前距离。如果当前距离小于包含源和其他顶点之间距离的字典中提到的距离，我们将更新字典中的距离。
*   在全宽度优先遍历之后，我们将得到包含从源到每个顶点的最小距离的字典。

我们可以将用于计算未加权图的顶点之间的最短路径长度的算法公式化如下。

1.  创建一个空队列。
2.  创建一个 visited_vertices 列表来跟踪已访问的顶点。
3.  创建一个字典 distance_dict 来跟踪顶点到源顶点的距离。将距离初始化为 99999999。
4.  将源顶点插入 Q 和 visited_vertices。
5.  如果 Q 为空，则返回。否则转到 6。
6.  从 q 中取出一个顶点 v。
7.  更新 distance_dict 中 v 的未访问邻居的距离。
8.  将未访问的相邻顶点插入 Q 和已访问的顶点。
9.  转到第 5 页。

## 履行

因为我们已经讨论了这个例子，并且制定了一个算法来寻找图中源顶点和其他顶点之间的最短路径长度，所以让我们用 python 来实现这个算法。

```py
from queue import Queue

graph = {'A': ['B', 'D', 'E', 'F'], 'D': ['A'], 'B': ['A', 'F', 'C'], 'F': ['B', 'A'], 'C': ['B'], 'E': ['A']}
print("Given Graph is:")
print(graph)

def calculate_distance(input_graph, source):
    Q = Queue()
    distance_dict = {k: 999999999 for k in input_graph.keys()}
    visited_vertices = list()
    Q.put(source)
    visited_vertices.append(source)
    while not Q.empty():
        vertex = Q.get()
        if vertex == source:
            distance_dict[vertex] = 0
        for u in input_graph[vertex]:
            if u not in visited_vertices:
                # update the distance
                if distance_dict[u] > distance_dict[vertex] + 1:
                    distance_dict[u] = distance_dict[vertex] + 1
                Q.put(u)
                visited_vertices.append(u)
    return distance_dict

distances = calculate_distance(graph, "A")
for vertex in distances:
    print("Shortest Path Length to {} from {} is {}.".format(vertex, "A", distances[vertex])) 
```

输出:

```py
Given Graph is:
{'A': ['B', 'D', 'E', 'F'], 'D': ['A'], 'B': ['A', 'F', 'C'], 'F': ['B', 'A'], 'C': ['B'], 'E': ['A']}
Shortest Path Length to A from A is 0.
Shortest Path Length to D from A is 1.
Shortest Path Length to B from A is 1.
Shortest Path Length to C from A is 2.
Shortest Path Length to E from A is 1.
```

## 结论

在本文中，我们讨论并实现了计算无权图中顶点间最短路径长度的算法。这里我们使用了广度优先图遍历算法。要了解二叉树遍历算法，可以阅读[中的有序树遍历算法](https://www.pythonforbeginners.com/data-structures/in-order-tree-traversal-in-python)或[中的层次有序树遍历算法](https://www.pythonforbeginners.com/data-structures/level-order-tree-traversal-in-python)。