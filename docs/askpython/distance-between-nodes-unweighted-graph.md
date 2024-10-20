# 计算未加权图中节点之间的距离

> 原文：<https://www.askpython.com/python/examples/distance-between-nodes-unweighted-graph>

图遍历算法有各种应用。其中一个应用是寻找图的两个节点之间的最小距离。在本文中，我们将使用广度优先图遍历算法在 python 中实现一个算法来寻找未加权全连通图中的最小距离。

## 对图使用 BFS 算法

[广度优先搜索](https://www.askpython.com/python/examples/breadth-first-search-graph)是一种图形遍历算法，其中我们从任意单个顶点开始，恰好遍历图形的每个顶点一次。对于每个选中的顶点，我们首先打印该顶点，然后打印它的所有邻居。这个过程一直持续到遍历完所有顶点。当使用广度优先搜索遍历图形时，看起来我们是从所选的顶点开始分层移动的。

图的 BFS 算法的实现如下。在这个算法中，我们已经假设该图是未加权的、无向的并且是完全连通的。

```py
def bfs(graph, source):
    Q = Queue()
    visited_vertices = set()
    Q.put(source)
    visited_vertices.update({0})
    while not Q.empty():
        vertex = Q.get()
        print(vertex, end="-->")
        for u in graph[vertex]:
            if u not in visited_vertices:
                Q.put(u)
                visited_vertices.update({u})

```

## 确定未加权图的两个节点之间的最小距离

我们可以使用广度优先搜索算法，通过对该算法进行某些修改来找到从一个源到所有节点的最小距离。

给定图的源和邻接表表示，我们将声明一个包含所有访问过的顶点的列表，我们还将创建一个字典，字典中的键确定顶点，值确定当前顶点和源之间的距离。

这里对 BFS 算法的修改将是，每当我们处理一个顶点 v 时，我们将更新它的邻居的距离。v 的邻居到源的距离等于 v 到源的距离加 1。

## 确定最小距离的算法

由于我们对如何确定从源到每个顶点的最小距离有一个大致的概念，我们将为其制定算法。

```py
Algorithm Least Distance:
Input: Graph(Adjacency list) and Source vertex
Output: A list with distance of each vertex from source 
Start:
    1.Create an empty queue Q.
    2.Create an empty set to keep record of visited vertices.
    3\. Create a dictionary in which keys of the dictionary determine the vertex and values determine the distance between current vertex and source.
    4.Insert source vertex into the Q and Mark the source as visited.
    5.If Q is empty, return. Else goto 6.
    6.Take out a vertex v from Q.
    7.Insert all the vertices in the adjacency list of v which are not in the visited list into Q and mark them visited after updating their distance from source.
    8.Goto 5.
Stop.

```

## 实现图的遍历来计算最小距离

由于我们已经制定了用于确定顶点与源的最小距离的算法，因此我们将实现该算法并对下图中给出的图表执行该算法。

![Graph Implementation In Python](img/5881ab97e34a7a44225b1fb5ded95f10.png)

Graph Implementation In Python- Askpython

该算法在 python 中的实现如下。

```py
from queue import Queue

myGraph = {0: [1, 3], 1: [0, 2, 3], 2: [4, 1, 5], 3: [4, 0, 1], 4: [2, 3, 5], 5: [4, 2]}

def leastDistance(graph, source):
    Q = Queue()
    # create a dictionary with large distance(infinity) of each vertex from source
    distance = {k: 9999999 for k in myGraph.keys()}
    visited_vertices = set()
    Q.put(source)
    visited_vertices.update({0})
    while not Q.empty():
        vertex = Q.get()
        if vertex == source:
            distance[vertex] = 0
        for u in graph[vertex]:
            if u not in visited_vertices:
                # update the distance
                if distance[u] > distance[vertex] + 1:
                    distance[u] = distance[vertex] + 1
                Q.put(u)
                visited_vertices.update({u})
    return distance

print("Least distance of vertices from vertex 0 is:")
print(leastDistance(myGraph, 0))

```

输出:

```py
Least distance of vertices from vertex 0 is:
{0: 0, 1: 1, 2: 2, 3: 1, 4: 2, 5: 3}

```

## 结论

在本文中，我们实现了一个算法，使用一个[深度优先搜索](https://www.askpython.com/python/examples/depth-first-search-in-a-graph)遍历算法找到一个源和一个图的其他顶点之间的最小距离。请继续关注更多内容丰富的文章。