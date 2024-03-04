# Python 中的树形数据结构

> 原文：<https://www.pythonforbeginners.com/data-structures/tree-data-structure-in-python>

就特性和数据结构而言，Python 是一种非常丰富的语言。它有很多内置的数据结构，比如 python 字典、列表、元组、集合、frozenset 等等。除此之外，我们还可以使用类创建自己的自定义数据结构。在本文中，我们将学习 Python 中的二叉树数据结构，并尝试使用一个例子来实现它。

## Python 中的树型数据结构是什么？

树是一种数据结构，其中数据项使用引用以分层方式连接。每棵树由一个根节点组成，从这个根节点我们可以访问树的每个元素。从根节点开始，每个节点包含零个或多个与其连接的节点作为子节点。一个简单的二叉树可以描述如下图所示。

![](img/2e7b5d8d30e468e7e8f0f2010f542fc3.png)



Tree Data Structure

## 树数据结构的一部分

树由根节点、叶节点和内部节点组成。每个节点通过一个称为边的引用连接到它的智利。

**根节点:**根节点是一棵树的最顶层节点。它总是创建树时创建的第一个节点，我们可以从根节点开始访问树的每个元素。在上面的例子中，包含元素 50 的节点是根节点。

**父节点:**任何节点的父节点都是引用当前节点的节点。在上面的示例中，50 是 20 和 45 的父级，20 是 11、46 和 15 的父级。同样，45 是 30 和 78 的父代。

**子节点:**父节点的子节点是父节点使用引用指向的节点。在上面的例子中，20 和 45 是 50 的孩子。节点 11、46 和 15 是 20 和 30 的子节点，节点 78 是 45 的子节点。

**边:**父节点通过其连接到子节点的引用称为边。在上面的例子中，连接任意两个节点的每个箭头都是一条边。

**叶节点:**这些是树中没有子节点的节点。在上面的例子中，11、46、15、30 和 78 是叶节点。

**内部节点:**内部节点是指至少有一个子节点的节点。在上面的例子中，50、20 和 45 是内部节点。

## 什么是二叉树？

二叉树是一种树形数据结构，其中每个节点最多可以有 2 个子节点。这意味着二叉树中的每个节点可以有一个、两个或没有子节点。二叉树中的每个节点都包含数据和对其子节点的引用。这两个孩子根据他们的位置被命名为左孩子和右孩子。二叉树中节点的结构如下图所示。

![Node of a Binary Tree in Python](img/cc8380b5ca553b34fbc050578e3c4385.png)



Node of a Binary Tree

我们可以使用如下的类在 python 中定义如上所示结构的节点。

```py
class BinaryTreeNode:
  def __init__(self, data):
    self.data = data
    self.leftChild = None
    self.rightChild=None
```

这里，节点的构造函数将数据值作为输入，创建 BinaryTreeNode 类型的对象，将数据字段初始化为给定的输入，并将对子节点的引用初始化为 None。稍后可以将子节点分配给节点。下图显示了一个二叉树的例子。

![ Binary Tree in Python](img/2ba371d6eab6e2644167dea0752b2af3.png)



Binary Tree Data Structure

我们可以用 python 实现上面的二叉树，如下所示。

```py
class BinaryTreeNode:
    def __init__(self, data):
        self.data = data
        self.leftChild = None
        self.rightChild = None

node1 = BinaryTreeNode(50)
node2 = BinaryTreeNode(20)
node3 = BinaryTreeNode(45)
node4 = BinaryTreeNode(11)
node5 = BinaryTreeNode(15)
node6 = BinaryTreeNode(30)
node7 = BinaryTreeNode(78)

node1.leftChild = node2
node1.rightChild = node3
node2.leftChild = node4
node2.rightChild = node5
node3.leftChild = node6
node3.rightChild = node7

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
45
Node is:
20
left child of the node is:
11
right child of the node is:
15
Node is:
45
left child of the node is:
30
right child of the node is:
78
Node is:
11
left child of the node is:
None
right child of the node is:
None
Node is:
15
left child of the node is:
None
right child of the node is:
None
Node is:
30
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

## 结论

在本文中，我们讨论了 Python 中的树形数据结构和二叉树数据结构。要了解更多关于 Python 中数据结构的知识，可以阅读这篇关于 python 中[链表的文章。](https://www.pythonforbeginners.com/lists/linked-list-in-python)