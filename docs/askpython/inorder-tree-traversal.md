# Python 中的有序树遍历[实现]

> 原文：<https://www.askpython.com/python/examples/inorder-tree-traversal>

在本文中，我们将研究有序树遍历的概念和算法。然后我们将在 python 中实现有序遍历的算法，并在一个[二叉查找树](https://www.askpython.com/python/examples/binary-search-tree)上运行它。

## 什么是有序树遍历？

有序遍历是一种深度优先的树遍历算法。在深度优先遍历中，我们从根节点开始，然后探索树的一个分支，直到结束，然后我们回溯并遍历另一个分支。

在有序遍历中，首先，我们遍历当前节点的左子树或左子树，然后我们遍历当前节点，然后我们遍历当前节点的右子树或右子树。我们递归地执行这个操作，直到所有的节点都被遍历。我们使用 inorder 遍历以升序打印二叉查找树的元素。

## 有序树遍历算法

下面是有序遍历的算法。

```py
Algorithm inorder:
Input: Reference to Root Node
Output:Prints All the nodes of the tree
Start.
1.If root is empty,return.
2.Traverse left subtree of the root.// inorder(root.leftChild)
3\. Traverse the root node. //print value at node
4\. Traverse the right subtree of the root.// inorder(root.rightChild)
End.

```

## 有序遍历算法在 Python 中的实现

现在，我们将实现上述算法，以在有序遍历中打印以下二叉查找树的节点。

![bst](img/84488fba58a7c4dae9beaca7b846692c.png)

Binary Search Tree

在下面的代码中，首先创建了上面的二叉查找树，然后输出二叉树的有序遍历。

```py
class BinaryTreeNode:
  def __init__(self, data):
    self.data = data
    self.leftChild = None
    self.rightChild=None

def insert(root,newValue):
    #if binary search tree is empty, make a new node and declare it as root
    if root is None:
        root=BinaryTreeNode(newValue)
        return root

    #binary search tree is not empty, so we will insert it into the tree
    #if newValue is less than value of data in root, add it to left subtree and proceed recursively
    if newValue<root.data:
        root.leftChild=insert(root.leftChild,newValue)
    else:
        #if newValue is greater than value of data in root, add it to right subtree and proceed recursively
        root.rightChild=insert(root.rightChild,newValue)
    return root

def inorder(root):
#if root is None,return
        if root==None:
            return
#traverse left subtree
        inorder(root.leftChild)
#traverse current node
        print(root.data)
#traverse right subtree
        inorder(root.rightChild)     

root= insert(None,15)
insert(root,10)
insert(root,25)
insert(root,6)
insert(root,14)
insert(root,20)
insert(root,60)
print("Printing values of binary search tree in Inorder Traversal.")
inorder(root)

```

输出:

```py
Printing values of binary search tree in Inorder Traversal.
6
10
14
15
20
25
60

```

在这里，我们可以看到值是以递增的顺序打印的。因此，如果要求您按升序打印二叉查找树中的数据，您只需对二叉查找树执行一次有序遍历。

## 结论

在本文中，我们学习了有序树遍历的概念。我们还研究了该算法，并用 python 实现了该算法来遍历二叉查找树，并发现对于二叉查找树，按顺序遍历会按升序打印值。请继续关注更多内容丰富的文章。

快乐学习！