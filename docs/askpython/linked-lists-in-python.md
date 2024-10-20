# Python 中的链表

> 原文：<https://www.askpython.com/python/examples/linked-lists-in-python>

Python 中的链表是最有趣的抽象数据类型之一，自 C/C++时代以来一直很流行。在本文中，我们将从头开始学习如何用 Python 实现一个链表。

## 什么是链表？

***链表*** 是一种线性数据结构，其中每个元素都是一个单独的对象。与数组不同，链表的元素不一起存储在内存中。

链表的每个元素指向它后面的元素。这里的点意味着每个元素**存储下一个元素的地址。**

在遍历链表时，我们使用这些指针从一个节点跳到下一个节点。

对于每个链表，有两个元素需要考虑:

*   节点——我们通过创建一个节点[类](https://www.askpython.com/python/oops/python-classes-objects)来处理它们
*   节点之间的连接——我们将用 Python 中的变量和[列表来处理这个问题。](https://www.askpython.com/python/difference-between-python-list-vs-array)

## 如何用 Python 创建链表？

让我们回顾一下用 Python 创建链表的步骤。

## 创建节点类

为了创建我们自己的链表，我们需要定义一个节点类。在我们定义一个节点类之前，我们需要考虑这个类应该有哪些字段。

一个链表节点应该存储两件事。这些是:

1.  **数据**
2.  **下一节点的地址**

让我们用这两个字段定义一个节点类。

```py
class Node(object):

    def __init__(self, data=None, next_node=None):
        self.data = data
        self.next_node = next_node

```

## 创建链表类

让我们创建另一个类，它将初始化一个空节点来创建一个链表。

```py
class LinkedList(object):
    def __init__(self, head=None):
        self.head = head

```

这门课中有几个重要的功能。让我们回顾一下每个类的目的和定义。

### 1.打印列表的功能

让我们写一个函数来打印我们的链表。为了[打印](https://www.askpython.com/python/built-in-methods/python-print-function)链表，我们需要遍历整个链表并不断打印每个节点的数据。

```py
def printList(self): 
        temp = self.head 
        while (temp): 
            print (temp.data, " -> ", end = '') 
            temp = temp.next_node
        print("")

```

### 2.获取列表的大小

让我们写一个返回链表大小的函数。为了计算大小，我们需要遍历整个列表，并在这样做的同时保存一个计数器。

```py
def size(self):
     current = self.head
     count = 0
     while current:
        count += 1
        current = current.next_node
     return count

```

这个函数将返回链表的大小。

### 3.在开头插入一个新节点

让我们写一个函数在头部插入一个新的节点。

```py
def insert_at_head(self, data):
      new_node = Node(data)
      new_node.next_node = self.head
      self.head = new_node

```

这将创建一个包含数据的新节点，并将其添加到 head 之前。然后，它将链表的头部指向这个新节点。

### 4.获取下一个节点

获取下一个节点的函数如下所示:

```py
 def get_next_node (self,node):
      return node.next_node.data

```

## 创建新的链接列表

让我们编写 main 函数，并使用上面创建的类创建一个链表。

```py
llist = LinkedList() 

```

这行代码用一个空节点初始化 llist [对象](https://www.askpython.com/python/oops/python-classes-objects)。

### 1.添加节点

让我们给这个节点添加一些数据。

```py
llist.head = Node(1)

```

为链表创建几个其他节点。

```py
 s = Node(2)
 t = Node(3) 

```

### 2.在节点之间创建链接

在单个节点之间创建链接是创建链表最重要的部分。

您可以使用以下方式创建链接:

```py
llist.head.next_node = s
s.next_node = t

```

### 3.打印列表中的节点

要验证列表是否创建成功，我们可以使用打印功能。

```py
llist.printList()

```

输出:

```py
1  -> 2  -> 3 

```

### 4.输出列表的大小

要输出列表的大小，调用我们上面写的 size 函数。

```py
 print(llist.size())

```

输出:

```py
3

```

### 5.插入新节点

让我们尝试使用上面的函数在链表的头部插入一些数据。

```py
llist.insert_at_head(5)

```

我们可以打印清单进行验证。

```py
llist.printList()

```

输出:

```py
5  -> 1  -> 2  -> 3

```

### 6.获取下一个节点

要获取下一个节点:

```py
print(llist.get_next_node(s))

```

输出:

```py
3

```

## Python 中链表的完整实现

完整的实现如下所示:

```py
class Node(object):

    def __init__(self, data=None, next_node=None):
        self.data = data
        self.next_node = next_node

class LinkedList(object):
    def __init__(self, head=None):
        self.head = head

    def size(self):
     current = self.head
     count = 0
     while current:
        count += 1
        current = current.next_node
     return count

    def printList(self): 
        temp = self.head 
        while (temp): 
            print (temp.data, " -> ", end = '') 
            temp = temp.next_node
        print("")

    def insert_at_head(self, data):
      new_node = Node(data)
      new_node.next_node = self.head
      self.head = new_node

    def get_next_node (self,node):
      return node.next_node.data

if __name__=='__main__': 

    llist = LinkedList() 

    llist.head = Node(1) 
    s = Node(2) 
    t = Node(3) 
    llist.head.next_node = s;
    s.next_node = t
    llist.printList()
    print(s.data)
    print(llist.size())
    print(llist.get_next_node(s))
    llist.insert_at_head(5)
    llist.printList()

```

## 结论

本教程讲述了 Python 中链表的实现。我们从头开始创建了自己的链表，并编写了一些额外的函数来打印列表，获取列表的大小，并在头部进行插入。