# 求二叉树中所有节点的和

> 原文：<https://www.askpython.com/python/examples/sum-of-all-nodes-binary-tree>

在本文中，我们将使用该算法来查找二叉树中所有节点的总和。我们已经讨论过 Python 中的[级顺序二叉树遍历。](https://www.askpython.com/python/examples/level-order-binary-tree)

## 如何求二叉树中所有节点的和？

为了找到二叉树中所有节点的总和，我们将遍历二叉树的每个节点并找到它们的总和。在本文中，我们将使用一种改进的层次顺序树遍历算法来查找所有节点的总和。对于这个任务，我们将维护一个变量来保存总和，在处理每个节点之后，我们将把它的值加到总和中。

例如，下面的二叉树的元素之和是 150。

![Askpython](img/dff3c2eff5a4472b438a8d43bc3f5a6f.png)

Binary tree

## 一种算法，用于计算二叉树中所有节点的和

如前所述，我们将使用层次顺序树遍历算法来制定算法，以找到二叉树的所有元素的总和。该算法可以用公式表示如下。该算法将二叉树的根作为输入，并将所有元素的和作为输出。

1.  如果根为空，则返回。
2.  设 Q 为队列。
3.  将总和初始化为 0。
4.  在 q 中插入 root。
5.  从 q 中取出一个节点。
6.  如果节点为空，请转到 10。否则，转到 7。
7.  将节点中的元素添加到 sum 中。
8.  将节点的左子节点插入 q。
9.  将节点的右子节点插入 q。
10.  检查 Q 是否为空。如果 Q 不为空，则转到 5。

## 算法在 Python 中的实现

正如我们已经讨论过的算法，我们将用 Python 实现该算法，并在上图给出的二叉树上执行它。

```py
from queue import Queue

class BinaryTreeNode:
    def __init__(self, data):
        self.data = data
        self.leftChild = None
        self.rightChild = None

def insert(root, newValue):
    # if binary search tree is empty, make a new node and declare it as root
    if root is None:
        root = BinaryTreeNode(newValue)
        return root
    # binary search tree is not empty, so we will insert it into the tree
    # if newValue is less than value of data in root, add it to left subtree and proceed recursively
    if newValue < root.data:
        root.leftChild = insert(root.leftChild, newValue)
    else:
        # if newValue is greater than value of data in root, add it to right subtree and proceed recursively
        root.rightChild = insert(root.rightChild, newValue)
    return root

def sumOfNodes(root):
    if root is None:
        return 0
    Q = Queue()
    Q.put(root)
    current_sum = 0
    while not Q.empty():
        node = Q.get()
        if node is None:
            continue
        current_sum = current_sum + node.data
        Q.put(node.leftChild)
        Q.put(node.rightChild)
    return current_sum

root = insert(None, 15)
insert(root, 10)
insert(root, 25)
insert(root, 6)
insert(root, 14)
insert(root, 20)
insert(root, 60)
print("Printing the sum of all the elements of the binary tree.")
print(sumOfNodes(root))

```

输出:

```py
Printing the sum of all the elements of the binary tree.
150

```

## 结论

在本文中，我们已经讨论了寻找二叉树所有元素之和的算法。请继续关注更多关于 Python 中不同算法实现的文章。