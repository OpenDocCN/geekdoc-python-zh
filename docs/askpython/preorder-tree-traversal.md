# Python 中的前序树遍历

> 原文：<https://www.askpython.com/python/examples/preorder-tree-traversal>

在本文中，我们将研究前序树遍历的概念和算法。然后我们将在 Python 中实现前序遍历算法，并在一棵[二叉树](https://www.askpython.com/python/examples/binary-tree-implementation)上运行它。

## 什么是前序树遍历？

前序遍历是一种深度优先的树遍历算法。在深度优先遍历中，我们从根节点开始，然后探索树的一个分支，直到结束，然后我们回溯并遍历另一个分支。

在前序遍历中，首先我们遍历当前节点，然后我们遍历当前节点的左子树或左子树，然后我们遍历当前节点的右子树或右子树。我们递归地执行这个操作，直到所有的节点都被遍历。

我们使用前序遍历来创建一个二叉树的副本。我们还可以使用前序遍历从表达式树中导出前缀表达式。

## Python 中的前序树遍历算法

下面是前序树遍历的算法。

算法预定–

*   输入:对根节点的引用
*   输出:打印树的所有节点
*   开始吧。
*   如果根为空，则返回。
*   遍历根节点。//打印节点的值
*   遍历根的左子树。// preorder(root.leftChild)
*   遍历根的右边子树。// preorder(root.rightChild)
*   结束。

## Python 中前序遍历算法的实现

现在我们将实现上面的算法来打印下面的二叉树的节点。

![Askpython31 1](img/dff3c2eff5a4472b438a8d43bc3f5a6f.png)

Binary Tree

在下面的代码中，首先创建了上面的二叉树，然后打印二叉树的前序遍历。

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
def preorder(root):
    #if root is None return
        if root==None:
            return
        #traverse root
        print(root.data)
        #traverse left subtree
        preorder(root.leftChild)
        #traverse right subtree
        preorder(root.rightChild)                   
root= insert(None,15)
insert(root,10)
insert(root,25)
insert(root,6)
insert(root,14)
insert(root,20)
insert(root,60)
print("Printing values of binary tree in preorder Traversal.")
preorder(root)

```

输出:

```py
Printing values of binary tree in preorder Traversal.
15
10
6
14
25
20
60

```

## 结论

在本文中，我们学习了前序树遍历的概念。我们还研究了该算法，并用 python 实现了它来遍历二叉树。请继续关注更多内容丰富的文章。

快乐学习。