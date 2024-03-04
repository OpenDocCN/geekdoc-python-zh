# Python 中的深度优先遍历

> 原文：<https://www.pythonforbeginners.com/data-structures/depth-first-traversal-in-python>

图形是非线性数据结构，用于表示不同对象之间的关系。在本文中，我们将讨论深度优先遍历算法来打印图中的顶点。我们还将在 python 中实现深度优先遍历算法，以说明该算法的工作原理。

## 什么是深度优先遍历？

深度优先遍历是一种图遍历算法，我们从图的一个顶点开始，打印出它的值。然后我们移动到当前顶点的一个邻居并打印它的值。如果当前顶点没有必须打印的相邻顶点，我们就移动到前一个顶点，看看是否打印了它们的所有相邻顶点。如果没有，我们选择一个邻居并打印它的值。我们重复这个过程，直到图形的所有顶点都打印出来。应该注意的是，我们只需打印每个顶点一次。

让我们在下图中尝试这个过程。

![Graph in Python](img/cc52d4090791fe078b53f6ba287546fb.png)



Graph

这里，让我们从顶点 a 开始。首先，我们将打印 a。之后，我们将从 B、D、E 和 F 中选择一个顶点，因为它们都是 a 的邻居。

让我们选择 B。在打印 B 之后，我们将从 A、C 和 F 中选择一个顶点，因为它们是 B 的邻居。这里，A 已经被打印，所以不会被选择。

让我们选择 C。在打印 C 之后，我们没有 C 的邻居要打印。因此，我们将移动到 C 的前一个顶点，即顶点 B，以检查是否已经打印了 B 的所有邻居。这里，F 还没有打印出来。所以我们会选择 f。

打印 F 之后，我们必须从 A 和 B 中选择一个顶点，因为它们是 F 的邻居，但是这两个顶点都已经打印出来了。因此，我们将移动到前一个顶点。

在 B 处，我们可以看到 B 的所有邻居都已经被打印了。因此，我们将移动到 B 的前一个顶点，即 a。

在 A 处，邻居 D 和 E 尚未打印，因此我们将选择其中一个顶点。

让我们选择 D。在打印 D 之后，我们可以看到 D 没有任何邻居要打印。因此，我们将移动到它的前一个顶点，即 a。

现在，只有 A 的一个邻居没有被打印，也就是说，我们将打印 e。

此时，图形的所有顶点都已按照 A、B、C、F、D、e 的顺序打印出来。因此，我们将停止这一过程。

您可以观察到，基于我们选择的邻居，单个图可能有许多深度优先遍历。

## 深度优先遍历算法

使用一个[堆栈数据结构](https://www.pythonforbeginners.com/data-types/stack-in-python)来实现图的深度优先遍历算法。这里，我们假设我们有一个连通图。换句话说，我们可以从起始顶点到达图的每个顶点。

我们将维护一个堆栈来存储未打印的顶点，并维护一个列表来存储已访问的顶点。之后，我们将使用以下算法处理该图。

1.  创建一个空栈来存储没有被打印的顶点。
2.  创建一个空列表 L 来存储访问过的顶点。
3.  将源顶点插入到 s 中，同样，将源顶点插入到 l 中。
4.  如果堆栈 S 为空，转到 9，否则转到 5。
5.  从 s 中取出一个顶点 v。
6.  打印以 v 为单位的值。
7.  将 v 的所有未访问的邻居插入堆栈 S 和列表 l。
8.  转到第 4 节。
9.  停下来。

可以使用上图中给出的图形的源代码来演示该算法。

```py
graph = {'A': ['B', 'D', 'E', 'F'], 'D': ['A'], 'B': ['A', 'F', 'C'], 'F': ['B', 'A'], 'C': ['B'], 'E': ['A']}
print("Given Graph is:")
print(graph)

def DFS_Algorithm(input_graph, source):
    stack = list()
    visited_list = list()
    print("At {}, adding vertex {} to stack and visited list".format(source, source))
    stack.append(source)
    visited_list.append(source)
    while stack:
        vertex = stack.pop()
        print("At vertex :", vertex)
        print("Printing vertex:", vertex)
        for u in input_graph[vertex]:
            if u not in visited_list:
                print("At {}, adding vertex {} to stack and visited list".format(vertex, u))
                stack.append(u)
                visited_list.append(u)
        print("Vertices in visited list are:", visited_list)
        print("Vertices in stack are:", stack)

print("Explanation of DFS traversal of graph with source A is:")
DFS_Algorithm(graph, "A") 
```

输出:

```py
Given Graph is:
{'A': ['B', 'D', 'E', 'F'], 'D': ['A'], 'B': ['A', 'F', 'C'], 'F': ['B', 'A'], 'C': ['B'], 'E': ['A']}
Explanation of DFS traversal of graph with source A is:
At A, adding vertex A to stack and visited list
At vertex : A
Printing vertex: A
At A, adding vertex B to stack and visited list
At A, adding vertex D to stack and visited list
At A, adding vertex E to stack and visited list
At A, adding vertex F to stack and visited list
Vertices in visited list are: ['A', 'B', 'D', 'E', 'F']
Vertices in stack are: ['B', 'D', 'E', 'F']
At vertex : F
Printing vertex: F
Vertices in visited list are: ['A', 'B', 'D', 'E', 'F']
Vertices in stack are: ['B', 'D', 'E']
At vertex : E
Printing vertex: E
Vertices in visited list are: ['A', 'B', 'D', 'E', 'F']
Vertices in stack are: ['B', 'D']
At vertex : D
Printing vertex: D
Vertices in visited list are: ['A', 'B', 'D', 'E', 'F']
Vertices in stack are: ['B']
At vertex : B
Printing vertex: B
At B, adding vertex C to stack and visited list
Vertices in visited list are: ['A', 'B', 'D', 'E', 'F', 'C']
Vertices in stack are: ['C']
At vertex : C
Printing vertex: C
Vertices in visited list are: ['A', 'B', 'D', 'E', 'F', 'C']
Vertices in stack are: [] 
```

在输出中，您可以观察到所有顶点最终都出现在已访问列表中，并且它们按照 A、F、E、D、B、c 的顺序打印。您可以看到顶点的顺序与我们在上一节中讨论的顺序不同。这是因为当我们选择一个顶点的邻居时，我们有许多选项可以选择，并且顺序取决于我们选择的顶点。

## 深度优先遍历在 Python 中的实现

我们已经讨论了图的深度优先遍历的一般思想，并观察了该算法如何使用 python 程序工作，我们可以如下实现深度优先遍历算法。

```py
graph = {'A': ['B', 'D', 'E', 'F'], 'D': ['A'], 'B': ['A', 'F', 'C'], 'F': ['B', 'A'], 'C': ['B'], 'E': ['A']}
print("Given Graph is:")
print(graph)

def dfs_traversal(input_graph, source):
    stack = list()
    visited_list = list()
    stack.append(source)
    visited_list.append(source)
    while stack:
        vertex = stack.pop()
        print(vertex, end=" ")
        for u in input_graph[vertex]:
            if u not in visited_list:
                stack.append(u)
                visited_list.append(u)

print("DFS traversal of graph with source A is:")
dfs_traversal(graph, "A")
```

输出:

```py
Given Graph is:
{'A': ['B', 'D', 'E', 'F'], 'D': ['A'], 'B': ['A', 'F', 'C'], 'F': ['B', 'A'], 'C': ['B'], 'E': ['A']}
DFS traversal of graph with source A is:
A F E D B C 
```

## 结论

在本文中，我们讨论了全连通图的深度优先遍历算法，并且用 Python 实现了它。在我们的实现中，我们使用了一个列表作为堆栈，但是堆栈也可以使用 Python 中的链表来实现。要了解更多关于其他算法的内容，您可以阅读这篇关于 Python 中的[有序树遍历的文章。](https://www.pythonforbeginners.com/data-structures/in-order-tree-traversal-in-python)