# 如何反转一个单链表

> 原文：<https://www.pythoncentral.io/reverse-singly-linked-list/>

## 先决条件

要学习如何反转单链表，你应该知道:

1.  Python 3
2.  Python 数据结构-列表
3.  [到位清单冲正](https://www.pythoncentral.io/python-reverse-list-place/)
4.  OOP 概念
5.  第一部分和第二部分单链表

## 我们会学到什么？

在上一个教程中，我们讨论了什么是单链表，[如何添加一个节点](https://www.pythoncentral.io/singly-linked-list-insert-node/)，[如何打印所有节点](https://www.pythoncentral.io/singly-linked-list-insert-node/)，[如何删除一个节点](https://www.pythoncentral.io/find-remove-node-linked-lists/)。如果您还没有阅读这些内容，我们强烈建议您先阅读，因为我们将基于这些概念进行构建。

本教程讲述了如何反转一个链表。 正如在之前的教程中所讨论的[，你可以通过交换最后一个和第一个值来执行原地反转，依此类推。但是这里我们要讨论一种不同的方法。这个想法是颠倒链接。于是 4 - > 2 - > 3(人头指向 4，3 点指向*无*)就变成了 4 < - 2 < - 3(人头指向 3，4 点指向*无*)。这可以迭代和递归地完成。](https://www.pythoncentral.io/python-reverse-list-place/)

我们将跟踪三样东西:当前元素、上一个元素和下一个元素。这是因为一旦我们颠倒了前一个节点和当前节点之间的链接，我们就无法移动到当前的下一个节点。这就是为什么必须跟踪当前的下一个节点。让我们看一个例子:

| 链表 | 上一个 | 货币 | nex | 反转后 |
| (h)4 - > 2 - > 3(无) | 无 | 4 | 2 | (无)4 - > 2 - > 3 |
| (无)4 - > 2 - > 3 | 4 | 2 | 3 | (无)4 < - 2 - > 3 |
| (无)4 < - 2 - > 3 | 2 | 3 | 无 | (无)4 < - 2 < - 3 |
| (无)4 < - 2 < - 3 | 3 | 无 | 无 | (无)4 < - 2 < - 3(h) |

 **注:** 最后，我们将 *头* 指针指向上一个节点。

## 如何实现这一点？

既然你已经很好的掌握了链表反转，那我们就来看看相关的算法和代码吧。

### 迭代法

#### 算法

1.  设置为 *无**当前* 为 *头**下一个* 为下一个节点 *当前*
2.  遍历链表，直到 *当前* 为 *无* (这是循环的退出条件)
3.  每次迭代时，将 *当前* 的下一个节点设置为 *先前*
4.  然后，设置为 *当前**当前* 为 *下一个* 和 *下一个* 为其下一个节点(这是循环的迭代过程)
5.  一旦变成了，设置头指针指向 *前一个* 节点。

#### 代码

```py
def reverseList(list):

       #Initializing values
       prev = None
       curr = list.head
       nex = curr.getNextNode()

       #looping
       while curr:
           #reversing the link
           curr.setNextNode(prev)     

           #moving to next node      
           prev = curr
           curr = nex
           if nex:
               nex = nex.getNextNode()

       #initializing head
       list.head = prev
```

### 递归方法

#### 算法

1.  传递指针到此方法为 *节点* 。
2.  检查 *节点* 的下一个节点是否为 *无* :
    1.  如果是，这表明我们已经到达了链表的末尾。设置 *头* 指针指向本节点
    2.  如果没有，将 *的下一个节点节点* 传递给 *反向* 方法
3.  一旦到达最后一个节点，就会发生逆转。

#### 代码

```py
def reverse(self,node):

       if node.getNextNode() == None:
           self.head = node
           return
       self.reverse(node.getNextNode())
       temp = node.getNextNode()
       temp.setNextNode(node)
       node.setNextNode(None)
```

## 结论

也尝试使用原地反转来解决上述问题。反转是解决与链表相关的其他问题的基础，我们将在以后的教程中看到。快乐的蟒蛇！