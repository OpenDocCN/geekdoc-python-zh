# Python 中的后序树遍历算法

> 原文：<https://www.pythonforbeginners.com/data-structures/postorder-tree-traversal-algorithm-in-python>

二叉树在表示层次数据方面非常有用。在本文中，我们将讨论如何使用后序树遍历打印二叉树中的所有元素。我们还将在 python 中实现后序树遍历算法。

## 后序树遍历算法是什么？

后序遍历算法是一种深度优先遍历算法。这里，我们从一个根节点开始，遍历树的一个分支，直到到达分支的末端。之后，我们转移到下一个分支。这个过程一直持续到树中的所有节点都被打印出来。

后序树遍历算法的名字来源于树的节点被打印的顺序。在这个算法中，我们首先打印节点的左子树，然后打印当前节点的右子树。最后，我们打印当前节点。这个过程本质上是递归的。这里，仅当当前节点的左子树和右子树中的所有节点都已经被打印时，才打印该节点。

让我们使用下图中给出的二叉树来理解这个过程。

![Binary tree](img/5a1058d8bba73a50f0af39e7ff548969.png)



Binary Tree

让我们使用后序遍历算法打印上述二叉树中的所有节点。

*   我们将从节点 50 开始。在打印 50 之前，我们必须打印它的左子树和右子树。所以，我们将移动到 20。
*   在打印 20 之前，我们必须打印它的左子树和右子树。所以，我们将搬到 11 楼。
*   由于 11 没有孩子，我们将打印 11。此后，我们将移动到前一个节点，即 20。
*   由于 20 的左侧子树已经打印，我们将移动到 20 的右侧子树，即 22。
*   由于 22 没有孩子，我们将打印 22。此后，我们将移动到前一个节点，即 20。
*   因为已经打印了 20 的左子树和右子树。我们将打印 20，并将移动到其父节点，即 50。
*   此时，左侧 50 个子树已经被打印。因此，将打印其右边的子树。我们将搬到 53 楼。
*   在打印 53 之前，我们必须打印它的左子树和右子树。所以，我们要搬到 52 楼。
*   由于 52 没有子节点，我们将打印 52。此后，我们将移动到前一个节点，即 53。
*   因为已经打印了 53 的左侧子树，所以我们将移动到 53 的右侧子树，即 78。
*   由于 78 没有子节点，我们将打印 78。此后，我们将移动到前一个节点，即 53。
*   由于 53 的左子树和右子树都已打印，我们将打印 53，并将移动到其父节点，即 50。
*   此时，50 的左子树和右子树都已经打印出来了，所以，我们将打印 50。
*   由于树中的所有节点都已经被打印，我们将终止这个算法。

您可以看到，我们按照 11、22、20、52、78、53、50 的顺序打印了这些值。现在让我们为后序树遍历算法制定一个算法。

## 后序树遍历算法

由于您对整个过程有一个总体的了解，我们可以将后序树遍历的算法公式化如下。

1.  从根节点开始。
2.  如果根为空，则返回。
3.  递归遍历左边的子树。
4.  递归遍历右边的子树。
5.  打印根节点。
6.  停下来。

## Python 中后序树遍历的实现

我们已经理解了后序树遍历的算法及其工作原理，让我们实现该算法，并对上图中给出的二叉树执行该算法。

```py
class BinaryTreeNode:
    def __init__(self, data):
        self.data = data
        self.leftChild = None
        self.rightChild = None

def postorder(root):
    # if root is None,return
    if root is None:
        return
    # traverse left subtree
    postorder(root.leftChild)

    # traverse right subtree
    postorder(root.rightChild)
    # print the current node
    print(root.data, end=" ,")

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
print("Postorder traversal of the binary tree is:")
postorder(root) 
```

输出:

```py
Postorder traversal of the binary tree is:
11 ,22 ,20 ,52 ,78 ,53 ,50 ,
```

## 结论

在本文中，我们讨论并实现了后序树遍历算法。要了解更多关于其他树遍历算法的内容，可以阅读这篇关于 [Inorder tree 遍历算法](https://www.pythonforbeginners.com/data-structures/in-order-tree-traversal-in-python)或 [level order tree 遍历算法](https://www.pythonforbeginners.com/data-structures/level-order-tree-traversal-in-python)in python 的文章。