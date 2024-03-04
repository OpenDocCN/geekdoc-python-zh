# 蟒蛇皮二叉查找树

> 原文：<https://www.pythonforbeginners.com/data-structures/binary-search-tree-in-python>

您可以在程序中使用不同的数据结构，如 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)、列表、元组或集合。但是这些数据结构不足以在程序中实现层次结构。在这篇文章中，我们将学习二叉查找树数据结构，并将在 python 中实现它们，以便更好地理解。

## 什么是二叉树？

二叉树是一种树形数据结构，其中每个节点最多可以有 2 个子节点。这意味着二叉树中的每个节点可以有一个、两个或没有子节点。二叉树中的每个节点都包含数据和对其子节点的引用。这两个孩子根据他们的位置被命名为左孩子和右孩子。二叉树中节点的结构如下图所示。

![Binary tree node](img/aa7c04f57e70e52d8322a01ddfa3c2a2.png)



Node of a Binary Tree

我们可以用 python 实现一个二叉树节点，如下所示。

```py
class BinaryTreeNode:
  def __init__(self, data):
    self.data = data
    self.leftChild = None
    self.rightChild=None
```

## 什么是二叉查找树？

二叉查找树是一种二叉树数据结构，具有以下属性。

*   二叉查找树中没有重复的元素。
*   节点左边子节点的元素总是小于当前节点的元素。
*   节点的左子树包含比当前节点少的所有元素。
*   节点右边子节点的元素总是大于当前节点的元素。
*   节点右边的子树包含所有大于当前节点的元素。

以下是满足上述所有属性的二叉查找树的示例。

![Binary search tree in Python](img/227ebcd313c831a275ded35498ebdd07.png)



Binary search tree

现在，我们将在二叉查找树上实现一些基本操作。

## 如何在二叉查找树中插入元素？

我们将使用二分搜索法树的属性在其中插入元素。如果我们想在特定的节点插入一个元素，可能会出现三种情况。

1.  当前节点可以是空节点，即无。在这种情况下，我们将使用要插入的元素创建一个新节点，并将这个新节点分配给当前节点。
2.  要插入的元素可以大于当前节点处的元素。在这种情况下，我们将在当前节点的右子树中插入新元素，因为任何节点的右子树都包含比当前节点大的所有元素。
3.  要插入的元素可以小于当前节点处的元素。在这种情况下，我们将在当前节点的左子树中插入新元素，因为任何节点的左子树都包含小于当前节点的所有元素。

为了插入元素，我们将从根节点开始，并根据上面定义的规则将元素插入到二叉查找树中。在二叉查找树中插入元素的算法在 Python 中实现如下。

```py
class BinaryTreeNode:
    def __init__(self, data):
        self.data = data
        self.leftChild = None
        self.rightChild = None

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
node1 = root
node2 = node1.leftChild
node3 = node1.rightChild
node4 = node2.leftChild
node5 = node2.rightChild
node6 = node3.leftChild
node7 = node3.rightChild
print("Root Node is:")
print(node1.data)

print("left child of the node is:")
print(node1.leftChild.data)

print("right child of the node is:")
print(node1.rightChild.data)

print("Node is:")
print(node2.data)

print("left child of the node is:")
print(node2.leftChild.data)

print("right child of the node is:")
print(node2.rightChild.data)

print("Node is:")
print(node3.data)

print("left child of the node is:")
print(node3.leftChild.data)

print("right child of the node is:")
print(node3.rightChild.data)

print("Node is:")
print(node4.data)

print("left child of the node is:")
print(node4.leftChild)

print("right child of the node is:")
print(node4.rightChild)

print("Node is:")
print(node5.data)

print("left child of the node is:")
print(node5.leftChild)

print("right child of the node is:")
print(node5.rightChild)

print("Node is:")
print(node6.data)

print("left child of the node is:")
print(node6.leftChild)

print("right child of the node is:")
print(node6.rightChild)

print("Node is:")
print(node7.data)

print("left child of the node is:")
print(node7.leftChild)

print("right child of the node is:")
print(node7.rightChild) 
```

输出:

```py
Root Node is:
50
left child of the node is:
20
right child of the node is:
53
Node is:
20
left child of the node is:
11
right child of the node is:
22
Node is:
53
left child of the node is:
52
right child of the node is:
78
Node is:
11
left child of the node is:
None
right child of the node is:
None
Node is:
22
left child of the node is:
None
right child of the node is:
None
Node is:
52
left child of the node is:
None
right child of the node is:
None
Node is:
78
left child of the node is:
None
right child of the node is:
None
```

## 如何在二分搜索法树中搜索元素？

如您所知，二叉查找树不能有重复的元素，我们可以使用以下基于二分搜索法树属性的规则来搜索二叉查找树中的任何元素。我们将从根开始，并遵循这些属性

1.  如果当前节点为空，我们将说该元素不在二叉查找树中。
2.  如果当前节点中的元素大于要搜索的元素，我们将搜索其左子树中的元素，因为任何节点的左子树都包含小于当前节点的所有元素。
3.  如果当前节点中的元素小于要搜索的元素，我们将搜索其右子树中的元素，因为任何节点的右子树都包含所有大于当前节点的元素。
4.  如果当前节点的元素等于要搜索的元素，我们将返回 True。

基于上述属性在二叉查找树中搜索元素的算法在下面的程序中实现。

```py
class BinaryTreeNode:
    def __init__(self, data):
        self.data = data
        self.leftChild = None
        self.rightChild = None

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

def search(root, value):
    # node is empty
    if root is None:
        return False
    # if element is equal to the element to be searched
    elif root.data == value:
        return True
    # element to be searched is less than the current node
    elif root.data > value:
        return search(root.leftChild, value)
    # element to be searched is greater than the current node
    else:
        return search(root.rightChild, value)

root = insert(None, 50)
insert(root, 20)
insert(root, 53)
insert(root, 11)
insert(root, 22)
insert(root, 52)
insert(root, 78)
print("53 is present in the binary tree:", search(root, 53))
print("100 is present in the binary tree:", search(root, 100))
```

输出:

```py
53 is present in the binary tree: True
100 is present in the binary tree: False
```

## 结论

在这篇文章中，我们讨论了二分搜索法树及其性质。我们还用 Python 实现了在二叉查找树中插入元素和在二叉查找树中搜索元素的算法。要了解更多关于 Python 中数据结构的知识，可以阅读这篇关于 python 中[链表的文章。](https://www.pythonforbeginners.com/lists/linked-list-in-python)