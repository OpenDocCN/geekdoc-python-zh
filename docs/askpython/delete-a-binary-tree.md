# 如何在 Python 中删除二叉树？

> 原文：<https://www.askpython.com/python/examples/delete-a-binary-tree>

我们在之前的帖子中已经讨论过[二叉树](https://www.askpython.com/python/examples/binary-tree-implementation)和[二分搜索法树](https://www.askpython.com/python/examples/binary-search-tree)。在本文中，我们将制定一个算法来删除二叉树，而不会导致内存泄漏。我们还将用 Python 实现该算法。

## 什么是内存泄漏？

当我们把内存分配给一个变量而忘记删除它时，程序中就会发生内存泄漏。内存泄漏会导致程序终止时出现问题。因此，在删除对内存的引用之前，有必要删除一个分配。

Python 使用垃圾收集过程来处理这些错误，但是我们应该注意不要编写会导致程序内存泄漏的代码。这里我们将讨论一个删除整个二叉树而不导致内存泄漏的算法。

## 如何在不内存泄漏的情况下删除二叉树的节点？

要删除二叉树的元素，我们可以使用 del 语句来释放分配给每个节点的内存。此外，为了避免内存泄漏，我们必须在删除节点本身之前删除节点的子节点。通过这种方式，我们可以确保引用一个节点的变量在释放内存之前不会被删除。

为了遍历整个树，我们可以使用任何树遍历算法，比如按序、前序、层次序或后序树遍历算法。但是，我们需要在父节点之前遍历子节点，因为子节点必须在父节点之前删除，以避免内存泄漏。

在[后序树](https://www.askpython.com/python/examples/postorder-tree-traversal-in-python)遍历算法中，我们在访问父节点之前先遍历任意节点的子节点。因此，我们将使用后序树遍历来实现删除二叉树的算法。在下一节中，我们将修改后序树遍历算法来实现该算法。

## 删除二叉树的算法

如上所述，删除二叉树的算法可以用公式表示如下。

1.  从根开始。
2.  检查当前节点是否为 None，如果是，返回。否则转到 3。
3.  递归删除当前节点的左子节点。
4.  递归删除当前节点的右子节点。
5.  删除当前节点。

## 在 Python 中删除二叉树

由于我们已经讨论并制定了删除二叉树的算法，我们将用 python 实现它。我们还将对下图中给出的二叉树执行算法。在输出中，您可以验证在删除其父节点之前是否删除了每个节点。

![Delete a Binary Tree](img/dff3c2eff5a4472b438a8d43bc3f5a6f.png)

Binary tree

代码:

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

def deleteTree(root):
    if root:
        # delete left subtree
        deleteTree(root.leftChild)
        # delete right subtree
        deleteTree(root.rightChild)
        # traverse root
        print("Deleting Node:", root.data)
        del root

root = insert(None, 15)
insert(root, 10)
insert(root, 25)
insert(root, 6)
insert(root, 14)
insert(root, 20)
insert(root, 60)
print("deleting all the elements of the binary tree.")
deleteTree(root)

```

输出:

```py
deleting all the elements of the binary tree.
Deleting Node: 6
Deleting Node: 14
Deleting Node: 10
Deleting Node: 20
Deleting Node: 60
Deleting Node: 25
Deleting Node: 15

```

## 结论

在本文中，我们讨论了用一种改进的后序树遍历算法删除二叉树的算法。请继续关注更多关于 Python 中不同算法实现的文章。