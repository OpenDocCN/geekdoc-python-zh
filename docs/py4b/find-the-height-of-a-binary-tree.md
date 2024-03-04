# 求二叉树的高度

> 原文：<https://www.pythonforbeginners.com/data-structures/find-the-height-of-a-binary-tree>

就像我们找到一个列表的长度或者一个 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)中的条目数一样，我们可以找到一棵二叉树的高度。在这篇文章中，我们将制定一个算法来寻找二叉树的高度。我们还将用 python 实现该算法，并在给定的二叉树上执行。

## 二叉树的高度是多少？

二叉树的高度被定义为距离二叉树中节点所在的根节点的最大距离。二叉树的高度取决于节点的数量及其在树中的位置。如果一棵树有“n”个节点，它的高度可以是 log(n) + 1 到 n 之间的任何值。如果该树完全向左或向右倾斜，则二叉树的高度为 n。如果树中的节点适当分布，并且该树是完全二叉树，则它的高度为 log(n)+1。

例如，下面的二叉树有 7 个元素。具有 7 个元素的二叉树可以具有 log(7)+ 1 之间的任何高度，即 3 和 7。在我们的例子中，树的节点是适当分布的，树是完全平衡的。因此，树的高度是 3。

![height of a binary tree ](img/84fb334697b8bf226b768ca3d2373899.png)



binary tree

## 如何计算二叉树的高度？

为了计算二叉树的高度，我们可以计算左右子树的高度。子树高度的最大值可以通过加 1 来求树的高度。对于一个空根，我们可以说树的高度为零。类似地，单个节点的高度将被认为是 1。

## 求二叉树高度的算法

现在我们已经找到了一种方法来寻找高度的二叉树，我们将制定算法来寻找高度如下。

1.  如果我们找到一个空的根节点，我们会说树的高度是 0。
2.  否则，我们将递归地找到左子树和右子树的高度。
3.  找到左子树和右子树的高度后，我们将计算它们的最大高度。
4.  我们将在最大高度上加 1。这将是二叉树的高度。

## 算法在 Python 中的实现

现在我们已经理解并制定了算法，我们将用 Python 实现它。

```py
from queue import Queue

class BinaryTreeNode:
    def __init__(self, data):
        self.data = data
        self.leftChild = None
        self.rightChild = None

def height(root):
    if root is None:
        return 0
    leftHeight=height(root.leftChild)
    rightHeight=height(root.rightChild)
    max_height= leftHeight
    if rightHeight>max_height:
        max_height = rightHeight
    return max_height+1

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
print("Height of the binary tree is:")
print(height(root)) 
```

输出:

```py
Height of the binary tree is:
3
```

这里，我们创建了一个二叉树节点。然后，我们定义了向二叉树插入元素的函数。最后，我们用 Python 实现了求二叉树高度的算法。

## 结论

在本文中，我们实现了一个算法来寻找二叉树的高度。要了解更多关于其他数据结构的知识，可以阅读这篇关于 Python 中的[链表的文章。请继续关注更多关于用 Python 实现不同算法的文章。](https://www.pythonforbeginners.com/lists/linked-list-in-python)