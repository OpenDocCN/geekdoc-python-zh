# 用 Python 实现队列

> 原文：<https://www.pythonforbeginners.com/queue/implement-queue-in-python>

队列是一种数据结构，它遵循先进先出(FIFO)的顺序来访问元素。在一个队列中，我们只能访问首先被添加的当前元素。队列在诸如广度优先搜索、计算机资源共享和进程管理等应用中有许多用途。在本文中，我们将尝试用 python 实现带链表的队列数据结构。

## 使用链表在 Python 中实现队列

链表是一种线性数据结构，其中每个数据对象指向另一个对象。要用链表实现一个队列，我们必须从链表中插入和删除元素，这样首先添加的元素只能被访问。只有当我们在链表的末尾插入元素，并从链表的开头访问或删除元素时，这才是可能的。这样，最老的元素将位于链表的开始，并且可以被访问或删除。

为了在 python 中实现一个带链表的队列，我们将首先定义一个节点对象，该对象将包含当前元素，并指向紧接在它之后插入的节点。在 python 中，该节点可以按如下方式实现。

```py
class Node:
    def __init__(self,data):
        self.data=data
        self.next=None
```

当我们用链表实现一个队列时，它将有一个名为 front 的元素，它将引用链表中插入的最老的元素，即第一个元素。所有其他元素都可以通过它来访问。它还将包含一个名为 queueSize 的变量，该变量将包含队列的长度。我们可以用 python 实现一个空队列，如下所示。

```py
class Queue:
    def __init__(self):
        self.front=None
        self.queueSize=0
```

## 用 Python 实现队列中的入队操作

当我们在队列中插入一个元素时，这个操作叫做入队操作。为了在链表的帮助下实现队列中的入队操作，对于每次插入，我们将在链表的末尾添加要插入的元素。这样，最近的元素总是在链表的末尾，最老的元素总是在链表的前面。将元素插入队列后，我们还将获得队列的大小，并将大小增加 1。可以使用 python 中的链表实现入队操作，如下所示。

```py
def enQueue(self,data):
        temp=Node(data)
        if self.front is None:
            self.front=temp
            self.queueSize= self.queueSize+1
        else:
            curr=self.front
            while curr.next!=None:
                curr=curr.next
            curr.next=temp
            self.queueSize=self.queueSize+1
```

## 用 python 实现队列中的出列操作

当我们从队列中取出一个元素时，这个操作被称为出列操作。要从队列中取出的元素是队列中最老的元素。如我们所知，在入队操作期间，我们将最近的元素添加到链表的末尾，而最老的元素位于链表的开始，由队列的前端指向，因此我们将移除前端指向的元素，并将前端指向下一个最老的元素。在执行出列操作之前，我们将检查队列是否为空。如果队列为空，即 front 指向 None，我们将使用 [python try except](https://www.pythonforbeginners.com/error-handling/python-try-and-except) 引发一个异常，并显示一条消息:队列为空。我们可以如下实现出列操作。

```py
def deQueue(self):
        try:
            if self.front == None:
                raise Exception("Queue is Empty")
            else:
                temp=self.front
                self.front=self.front.next
                tempdata=temp.data
                self.queueSize= self.queueSize-1
                del temp
                return tempdata
        except Exception as e:
            print(str(e))
```

## 如何访问队列的前端元素

由于队列的前端元素被前端引用，我们只需返回前端指向的节点中的数据。它可以按如下方式实现。

```py
def front_element(self):
        try:
            if self.front == None:
                raise Exception("Queue is Empty")
            else:
                return self.front.data
        except Exception as e:
            print(str(e))
```

## 获取队列的大小

因为我们将队列的当前大小保存在 queueSize 中，所以我们只需返回队列的 queueSize 元素中的值。这可以如下进行。

```py
def size(self):
        return self.queueSize
```

## 检查队列是否为空

前面的元素是指队列中最老的元素。如果队列中没有元素，前端将指向 None。因此，要检查队列是否为空，我们只需检查前端是否指向 None。我们可以实现 isEmpty()方法来检查队列是否为空，如下所示。

```py
def isEmpty(self):
        if self.queueSize==0:
            return True
        else:
            return False
```

使用 python 中的链表实现队列的完整工作代码如下。

```py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 00:13:19 2021

@author: aditya1117
"""

class Node:
    def __init__(self,data):
        self.data=data
        self.next=None
class Queue:
    def __init__(self):
        self.front=None
        self.queueSize=0
    def enQueue(self,data):
        temp=Node(data)
        if self.front is None:
            self.front=temp
            self.queueSize= self.queueSize+1
        else:
            curr=self.front
            while curr.next!=None:
                curr=curr.next
            curr.next=temp
            self.queueSize=self.queueSize+1
    def deQueue(self):
        try:
            if self.front == None:
                raise Exception("Queue is Empty")
            else:
                temp=self.front
                self.front=self.front.next
                tempdata=temp.data
                self.queueSize= self.queueSize-1
                del temp
                return tempdata
        except Exception as e:
            print(str(e))
    def isEmpty(self):
        if self.queueSize==0:
            return True
        else:
            return False
    def size(self):
        return self.queueSize
    def front_element(self):
        try:
            if self.front == None:
                raise Exception("Queue is Empty")
            else:
                return self.front.data
        except Exception as e:
            print(str(e))
```

## 结论

在本文中，我们使用 python 中的链表实现了队列及其所有操作。为了更深入地了解它，并理解 queue 与内置的数据结构(如 [python dictionary](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/) 、list 和 set)有何不同，复制上面示例中给出的完整代码，将其粘贴到您的 IDE 中，并试验其中的操作。请继续关注更多内容丰富的文章。