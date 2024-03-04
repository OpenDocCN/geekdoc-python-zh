# 找到二叉树的镜像

> 原文：<https://www.pythonforbeginners.com/data-structures/find-the-mirror-image-of-a-binary-tree>

与 [Python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python)、列表或集合不同，二叉树的元素以分层的方式表示。二叉树中的层次结构允许人们找到它的镜像，因为二叉树中的每个元素都有固定的位置。在本文中，我们将研究寻找二叉树镜像的算法。我们还将用 Python 实现该算法，并将在一个示例二叉树上执行它。

## 二叉树的镜像是什么？

二叉树的镜像是可以通过在树的每个节点交换左子和右子来创建的另一个二叉树。因此，要找到二叉树的镜像，我们只需交换二叉树中每个节点的左子节点和右子节点。让我们试着找出下面这棵树的镜像。

![mirror image of a binary tree](img/fc25658c5c43d8ebd7acef3d10811c94.png)



a binary tree

为了找到上述树的镜像，我们将从根开始并交换每个节点的子节点。

在根，我们将交换二叉树的左右子树。这样，20、11 和 22 将进入二叉树的右边子树，53、52 和 78 将进入二叉树的左边，如下所示。

![](img/83beb1ed7c8252216e3f2762480de35e.png)



然后我们就进入下一关，交换 53 的孩子。在这里，78 将成为 53 的左孩子，而 52 将成为 53 的右孩子。同样，我们将交换 20 的左孩子和右孩子。这样，22 将成为 20 的左孩子，而 11 将成为 20 的右孩子。在这一级交换节点后的输出二叉树将如下所示。

![](img/2ae3d77c90c0f91067c30398fa49b451.png)



现在我们将进入下一个阶段。在这个级别，所有节点都是叶节点，没有子节点。因此，在这一级不会有交换，上面的图像是最终的输出。

## 寻找二叉树镜像的算法

正如我们在上面看到的，我们可以通过交换每个节点的左子节点和右子节点来找到二叉树的镜像。让我们试着用一种系统的方式来表述这个算法。

在最后一个例子中，在第二层，每个节点只有叶节点作为它们的子节点。为了找到这一层节点的镜像，我们交换了这一层每个节点的左右子节点。在根节点，我们交换了它的两个子树。当我们交换每个子树时(叶节点也是子树)，我们可以使用递归实现这个算法。

寻找二叉树的镜像的算法可以用公式表示如下。

1.  从根节点开始。
2.  递归查找左侧子树的镜像。
3.  递归查找右边子树的镜像。
4.  交换左右子树。

## Python 中算法的实现

现在我们将实现算法，用 Python 找到二叉树的镜像。

```py
from queue import Queue

class BinaryTreeNode:
    def __init__(self, data):
        self.data = data
        self.leftChild = None
        self.rightChild = None

def mirror(node):
    if node is None:
        return None
    mirror(node.leftChild)
    mirror(node.rightChild)
    temp = node.leftChild
    node.leftChild = node.rightChild
    node.rightChild = temp

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

def inorder(root):
    if root:
        inorder(root.leftChild)
        print(root.data, end=" ")
        inorder(root.rightChild)

root = insert(None, 50)
insert(root, 20)
insert(root, 53)
insert(root, 11)
insert(root, 22)
insert(root, 52)
insert(root, 78)
print("Inorder Traversal of tree before mirroring:")
inorder(root)
mirror(root)
print("\nInorder Traversal of tree after mirroring:")
inorder(root) 
```

输出:

```py
Inorder Traversal of tree before mirroring:
11 20 22 50 52 53 78 
Inorder Traversal of tree after mirroring:
78 53 52 50 22 20 11 
```

这里，我们创建了一个二叉树节点。之后，我们定义了向二叉树插入元素的函数。我们还使用有序树遍历算法在找到镜像之前和之后打印树的元素。

## 结论

在这篇文章中，我们实现了一个算法来寻找一个二叉树的镜像。要了解更多关于其他数据结构的知识，可以阅读这篇关于 Python 中的[链表的文章。请继续关注更多关于用 Python 实现不同算法的文章。](https://www.pythonforbeginners.com/lists/linked-list-in-python)