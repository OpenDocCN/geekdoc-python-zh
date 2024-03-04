# Python 中的前序树遍历算法

> 原文：<https://www.pythonforbeginners.com/data-structures/preorder-tree-traversal-algorithm-in-python>

二叉树在表示层次数据方面非常有用。在本文中，我们将讨论如何用 python 打印一棵[二叉树中的所有元素。为此，我们将使用前序树遍历算法。我们还将在 python 中实现前序树遍历。](https://www.pythonforbeginners.com/data-structures/tree-data-structure-in-python)

## 什么是前序树遍历？

前序树遍历是一种深度优先遍历算法。这里，我们从一个根节点开始，遍历树的一个分支，直到到达分支的末端。之后，我们转移到下一个分支。这个过程一直持续到树中的所有节点都被打印出来。

前序树遍历算法的名字来源于树的节点被打印的顺序。在这个算法中，我们首先打印一个节点。之后，我们打印节点的左边的子节点。最后，我们打印出节点的正确子节点。这个过程本质上是递归的。这里，只有当当前节点的左子树中的所有节点和当前节点本身都已经被打印时，才打印节点的右子节点。

让我们使用下图中给出的二叉树来理解这个过程。

![](img/fa8502948ae5dbb1bd07d78a9f60f85e.png)



Binary Tree

让我们使用前序遍历打印上述二叉树中的所有节点。

*   首先，我们将从根节点开始，打印它的值，即 50。
*   之后，我们必须打印 50 的左孩子。因此，我们将打印 20 个。
*   打印完 20，还要打印 20 的左子。因此，我们将打印 11。
*   11 没有孩子。因此，我们将移动到节点 20 并打印它的右边的子节点，即我们将打印 22。
*   22 没有孩子，所以我们将搬到 20。所有 20 岁的孩子都被打印出来了。所以，我们将移动到 50。在 50 处，我们将打印它的右子节点，因为它的左子树中的所有节点都已经被打印了。因此，我们将打印 53。
*   打印完 53，还要打印 53 的左子。因此，我们将打印 52。
*   52 没有孩子。因此，我们将移动到节点 53 并打印它的右子节点，即我们将打印 78。
*   此时，我们已经打印了二叉树中的所有节点。因此，我们将终止这一进程。

您可以看到，我们按照 50、20、11、22、53、52、78 的顺序打印了这些值。现在让我们为前序树遍历制定一个算法。

## 前序树遍历算法

由于您对整个过程有一个总体的了解，我们可以将前序树遍历的算法公式化如下。

1.  从根节点开始。
2.  如果根目录为空，转到 6。
3.  打印根节点。
4.  递归遍历左边的子树。
5.  递归遍历右边的子树。
6.  停下来。

## Python 中前序树遍历的实现

我们已经讨论了前序树遍历的算法及其工作原理，让我们实现该算法，并对上图中给出的二叉树执行该算法。

```py
class BinaryTreeNode:
    def __init__(self, data):
        self.data = data
        self.leftChild = None
        self.rightChild = None

def preorder(root):
    # if root is None,return
    if root is None:
        return
    # print the current node
    print(root.data, end=" ,")
    # traverse left subtree
    preorder(root.leftChild)

    # traverse right subtree
    preorder(root.rightChild)

def insert(root, newValue):
    # if binary search tree is empty, create a new node and declare it as root
    if root is None:
        root = BinaryTreeNode(newValue)
        return root
    # if newValue is less than value of data in root, add it to left subtree and proceed recursively
    if newValue < root.data:
        root.leftChild = insert(root.leftChild, newValue)
    else:
        # if newValue is greater than value of data in root, add it to right subtree and proceed recursively
        root.rightChild = insert(root.rightChild, newValue)
    return root

root = insert(None, 50)
insert(root, 20)
insert(root, 53)
insert(root, 11)
insert(root, 22)
insert(root, 52)
insert(root, 78)
print("Preorder traversal of the binary tree is:")
preorder(root) 
```

输出:

```py
Preorder traversal of the binary tree is:
50 ,20 ,11 ,22 ,53 ,52 ,78 ,
```

您可以观察到，代码给出的输出与我们在讨论该算法时得到的输出相同。

## 结论

在本文中，我们讨论并实现了 Python 中的前序树遍历算法。要了解其他树遍历算法的更多信息，可以阅读 python 中的顺序树遍历和 Python 中的层次顺序树遍历中关于[的文章。](https://www.pythonforbeginners.com/data-structures/in-order-tree-traversal-in-python)