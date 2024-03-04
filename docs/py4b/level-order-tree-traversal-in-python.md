# Python 中的层次顺序树遍历

> 原文：<https://www.pythonforbeginners.com/data-structures/level-order-tree-traversal-in-python>

就像我们遍历一个 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)、一个列表或元组来访问它的元素一样，我们也可以遍历二叉树来访问它们的元素。有四种树遍历算法，即按序树遍历、前序树遍历、后序树遍历和层次序树遍历。在本文中，我们将讨论层次顺序树遍历算法，并用 python 实现它。

## 什么是层次顺序树遍历？

层次顺序树遍历算法是一种广度优先的树遍历算法。这意味着在遍历树时，我们首先遍历当前层的所有元素，然后再移动到下一层。为了理解这个概念，让我们考虑下面的二叉查找树。

![Level order tree traversal in Python](img/4d8d9dc1206e982cf473a74a4868ac1c.png)



上述树的层次顺序遍历如下。

我们将从根开始，打印 50 个。之后，我们移动到下一个级别，打印 20 和 53。在这一层之后，我们移动到另一层，打印 11、22、52 和 78。因此，上述树的层次顺序遍历是 50，20，53，11，22，52，78。

## 层次顺序树遍历算法

正如我们已经看到的，我们必须在层次顺序树遍历中一个接一个地处理每个层次上的元素，我们可以使用下面的方法。我们将从根节点开始，并打印它的值。之后，我们将把根节点的两个子节点移动到一个队列中。该队列将用于包含接下来必须处理的元素。每当我们处理一个元素时，我们会将该元素的子元素放入队列中。这样，同一级别的所有元素将以连续的顺序被推入队列，并以相同的顺序被处理。

层级顺序树遍历的算法可以用公式表示如下。

1.  定义一个队列 Q 来包含元素。
2.  将根插入 q。
3.  从 q 中取出一个节点。
4.  如果节点为空，即无，转到 8。
5.  打印节点中的元素。
6.  将当前节点的左子节点插入 q。
7.  将当前节点的右子节点插入 q。
8.  检查 Q 是否为空。如果是，停止。否则，转到 3。

## 层次顺序树遍历算法在 Python 中的实现

现在我们将用 python 实现上述算法。之后，我们将使用相同的算法处理上述示例中使用的二叉查找树。

```py
from queue import Queue

class BinaryTreeNode:
    def __init__(self, data):
        self.data = data
        self.leftChild = None
        self.rightChild = None

def levelorder(root):
    Q = Queue()
    Q.put(root)
    while (not Q.empty()):
        node = Q.get()
        if node == None:
            continue
        print(node.data)
        Q.put(node.leftChild)
        Q.put(node.rightChild)

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
print("Level Order traversal of the binary tree is:")
levelorder(root) 
```

输出:

```py
Level Order traversal of the binary tree is:
50
20
53
11
22
52
78 
```

在上面的程序中，我们首先实现了图中给出的二叉查找树。然后，我们使用层次顺序树遍历算法来遍历 Python 中的二叉查找树。正如您所看到的，程序使用了一个队列来存储要处理的数据。此外，二叉查找树的元素按照它们在树中的深度顺序从左到右打印。首先打印根节点，最后打印叶节点。

## 结论

在本文中，我们讨论了 Python 中的层次顺序树遍历算法。层次顺序树遍历可以用来在 Python 中找到二叉树的宽度。此外，该算法还被用于实现其他各种算法，我们将在本系列的后面讨论这些算法。在接下来的文章中，我们将实现其他的树遍历算法，比如有序树遍历、前序树遍历和后序树遍历算法。要了解更多关于其他数据结构的知识，可以阅读这篇关于 Python 中的[链表的文章。请继续关注更多关于用 Python 实现不同算法的文章。](https://www.pythonforbeginners.com/lists/linked-list-in-python)