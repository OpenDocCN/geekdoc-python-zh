# 图中深度优先搜索

> 原文：<https://www.askpython.com/python/examples/depth-first-search-in-a-graph>

深度优先搜索是一种遍历技术，在这种技术中，我们遍历一个图并精确地打印一次顶点。在本文中，我们将研究并实现 python 中遍历图的深度优先搜索。

***推荐阅读:[用 Python 实现一个图](https://www.askpython.com/python/examples/graph-in-python)***

## 深度优先搜索算法是什么？

在深度优先搜索中，我们从任意一个顶点开始，遍历图中的每个顶点一次。对于每个选定的顶点，我们首先打印该顶点，然后移动到它的一个邻居，打印它，然后移动到它的一个邻居，依此类推。这个过程一直持续到遍历完所有顶点。当使用深度优先搜索遍历一个图时，看起来我们是在从所选顶点开始遍历所有顶点的路径上移动。

从下面的例子可以清楚地理解这一点。

![Depth-first search Graph Implementation In Python](img/5881ab97e34a7a44225b1fb5ded95f10.png)

Graph Implementation In Python- Askpython

如果我们从 0 开始以深度优先的方式访问上图，我们将按照 0 –> 3 –> 4 –> 5 –> 2 –> 1 的顺序处理顶点。也可能有替代遍历。如果我们在 3 之前处理 1，而我们在 0，那么图的 BFS 遍历将看起来像:0 –> 1 –> 3-> 4-> 2-> 5。

## 图的深度优先搜索算法

由于我们对深度优先搜索有了一个总体的概念，我们现在将为图的 DFS 遍历制定算法。这里，我们将假设图的所有顶点都可以从起始顶点到达。

假设我们已经得到了一个邻接表表示的图和一个起始顶点。现在我们必须以深度优先搜索的方式遍历图形。

我们将首先打印起始顶点的值，然后我们将移动到它的一个邻居，打印它的值，然后移动到它的一个邻居，等等，直到图形的所有顶点都被打印出来。

因此，我们的任务是打印图的顶点，从第一个顶点开始，直到每个顶点都按顺序遍历。为了实现这个概念，我们将使用后进先出技术，即堆栈来处理图形。此外，我们将使用一个访问过的顶点列表来检查顶点是否在过去被遍历过，这样就不会有顶点被打印两次。

我们将打印一个顶点，将其添加到访问过的顶点列表中，并将其邻居放入堆栈中。然后，我们会从栈中一个一个的取出顶点，打印出来后添加到访问过的列表中，然后我们会把它们的邻居放入栈中。下面是描述整个过程的图的深度优先搜索遍历算法。

```py
Algorithm DFS:
Input: Graph(Adjacency list) and Source vertex
Output: DFS traversal of graph
Start:
    1.Create an empty stack S.
    2.Create an empty  list to keep record of visited vertices.
    3.Insert source vertex into S, mark the source as visited.
    4.If S is empty, return. Else goto 5.
    5.Take out a vertex v from S.
    6.Print the Vertex v.
    7.Insert all the unvisited vertices in the adjacency list of v into S and mark them visited.
    10.Goto 4.
Stop.

```

## 深度优先搜索遍历图在 python 中的实现

现在我们已经熟悉了这些概念和算法，我们将实现图的深度优先搜索算法，然后我们将执行上面例子中给出的图的算法。

```py
graph = {0: [1, 3], 1: [0, 2, 3], 2: [4, 1, 5], 3: [4, 0, 1], 4: [2, 3, 5], 5: [4, 2], 6: []}
print("The adjacency List representing the graph is:")
print(graph)

def dfs(graph, source):
    S = list()
    visited_vertices = list()
    S.append(source)
    visited_vertices.append(source)
    while S:
        vertex = S.pop()
        print(vertex, end="-->")
        for u in graph[vertex]:
            if u not in visited_vertices:
                S.append(u)
                visited_vertices.append(u)

print("DFS traversal of graph with source 0 is:")
dfs(graph, 0)

```

输出:

```py
The adjacency List representing the graph is:
{0: [1, 3], 1: [0, 2, 3], 2: [4, 1, 5], 3: [4, 0, 1], 4: [2, 3, 5], 5: [4, 2], 6: []}
DFS traversal of graph with source 0 is:
0-->3-->4-->5-->2-->1-->

```

如果你还不能理解代码的执行，这里有一个修改的 DFS 算法解释每一步。

```py
graph = {0: [1, 3], 1: [0, 2, 3], 2: [4, 1, 5], 3: [4, 0, 1], 4: [2, 3, 5], 5: [4, 2], 6: []}
print("The adjacency List representing the graph is:")
print(graph)

def dfs_explanation(graph, source):
    S = list()
    visited_vertices = list()
    S.append(source)
    visited_vertices.append(source)
    while S:
        vertex = S.pop()
        print("processing vertex {}.".format(vertex))
        for u in graph[vertex]:
            if u not in visited_vertices:
                print("At {}, adding {} to Stack".format(vertex, u))
                S.append(u)
                visited_vertices.append(u)
        print("Visited vertices are:", visited_vertices)

print("Explanation of DFS traversal of graph with source 0 is:")
dfs_explanation(graph, 0)

```

输出:

```py
The adjacency List representing the graph is:
{0: [1, 3], 1: [0, 2, 3], 2: [4, 1, 5], 3: [4, 0, 1], 4: [2, 3, 5], 5: [4, 2], 6: []}
Explanation of DFS traversal of graph with source 0 is:
processing vertex 0.
At 0, adding 1 to Stack
At 0, adding 3 to Stack
Visited vertices are: [0, 1, 3]
processing vertex 3.
At 3, adding 4 to Stack
Visited vertices are: [0, 1, 3, 4]
processing vertex 4.
At 4, adding 2 to Stack
At 4, adding 5 to Stack
Visited vertices are: [0, 1, 3, 4, 2, 5]
processing vertex 5.
Visited vertices are: [0, 1, 3, 4, 2, 5]
processing vertex 2.
Visited vertices are: [0, 1, 3, 4, 2, 5]
processing vertex 1.
Visited vertices are: [0, 1, 3, 4, 2, 5]

```

## 结论

在本文中，我们看到了图的深度优先搜索遍历算法背后的基本概念，设计了算法，然后用 python 实现了它。请继续关注更多内容丰富的文章。