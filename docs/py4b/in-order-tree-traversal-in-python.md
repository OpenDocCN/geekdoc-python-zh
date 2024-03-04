# Python 中的有序树遍历

> 原文：<https://www.pythonforbeginners.com/data-structures/in-order-tree-traversal-in-python>

你可能研究过遍历一个 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)、一个列表或者一个元组的算法。在本文中，我们将研究遍历二叉树的有序遍历算法。我们还将讨论算法的实现。

## 什么是有序树遍历算法？

有序树遍历算法是一种深度优先遍历算法。这意味着我们先遍历节点的子节点，然后再遍历它的兄弟节点。这样，我们遍历到最大深度，打印元素直到最后一个节点，然后回到顶部打印其他元素。

该算法被称为按序遍历算法，因为当我们使用该算法遍历二叉查找树时，树的元素按其值的升序打印。

## 有序遍历算法

我们知道，在二叉查找树中，任何节点的左子节点包含一个小于当前节点的元素，而节点的右子节点包含一个大于当前节点的元素。因此，为了按顺序打印元素，我们必须首先打印左边的子节点，然后是当前节点，最后是右边的子节点。这个相同的规则将用于打印树的每个节点。

我们将从根节点开始，遍历当前节点的左子树或左子树，然后遍历当前节点。最后，我们将遍历当前节点的右子树。我们将递归地执行这个操作，直到遍历完所有节点。

为了更好地理解这个概念，让我们使用一个示例二叉查找树，并使用有序二叉树遍历算法遍历它。

![In-order Tree Traversal in Python](img/227ebcd313c831a275ded35498ebdd07.png)



In-order Tree traversal

这里，我们将从根节点 50 开始。在打印 50 之前，我们必须遍历 50 的左子树。所以，我们要去 20 号。在打印 20 之前，我们必须遍历 20 的左子树。所以，我们要去 11 号。由于 11 没有子级，我们将打印 11 并向上移动到 20。打印完 20 之后，我们将遍历 20 的右子树。由于 22 没有孩子，我们将打印 22 并向上移动到 50。在 50，它的左子树已经被遍历，所以我们将打印 50。之后，我们将使用相同的过程遍历 50 的右子树。全树的有序遍历是:11，20，22，50，52，53，78。

## 有序树遍历在 Python 中的实现

有序树遍历的算法可以用公式表示如下。

1.  递归遍历左边的子树。
2.  打印根节点。
3.  递归遍历右边的子树。

这里，我们只打印节点中的元素，如果它是一个叶节点或者它的左子树中的所有元素都已经被打印。

我们现在将在 python 中实现上述算法，并对上述示例中给出的二叉查找树执行该算法。

```py
class BinaryTreeNode:
    def __init__(self, data):
        self.data = data
        self.leftChild = None
        self.rightChild = None

def inorder(root):
    # if root is None,return
    if root == None:
        return
    # traverse left subtree
    inorder(root.leftChild)
    # print the current node
    print(root.data, end=" ,")
    # traverse right subtree
    inorder(root.rightChild)

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
print("Inorder traversal of the binary tree is:")
inorder(root) 
```

输出:

```py
Inorder traversal of the binary tree is:
11 ,20 ,22 ,50 ,52 ,53 ,78 ,
```

## 结论

在本文中，我们讨论了 Python 中的有序树遍历算法。在接下来的文章中，我们将实现其他的树遍历算法，比如前序树遍历和后序树遍历算法。要了解更多关于其他数据结构的知识，可以阅读这篇关于 Python 中的[链表的文章。](https://www.pythonforbeginners.com/lists/linked-list-in-python)