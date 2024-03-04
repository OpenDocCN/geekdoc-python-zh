# 用 Python 实现 Deque

> 原文：<https://www.pythonforbeginners.com/deque/implement-deque-in-python>

双端队列是线性数据结构，我们可以用它来执行后进先出(LIFO)操作和先进先出(FIFO)操作。Deques 在现实生活中有很多应用，比如在软件中实现撤销操作，在 web 浏览器中存储浏览历史。在本文中，我们将使用链表在 python 中实现 deque。

## 如何在 python 中使用链表实现 deque？

链表是由包含数据的节点组成的线性数据结构。链表中的每个节点都指向链表中的另一个节点。为了使用[链表](https://www.pythonforbeginners.com/lists/linked-list-in-python)在 python 中实现 deque，我们必须在链表的两端执行插入和删除操作。此外，我们还必须记录队列的长度。

为了创建一个链表，我们将定义一个节点，它将有一个数据字段和下一个字段。数据字段包含实际数据，下一个字段引用链表中的下一个节点。python 中的节点类可以定义如下。

```py
class Node:
    def __init__(self,data):
        self.data=data
        self.next=None
```

为了使用链表实现 dequeue，我们将定义一个类 dequeue，它包含一个对链表第一个元素的名为 front 的引用和一个 dequeSize 字段来保持 dequeue 的大小，如下所示。

```py
class Deque:
    def __init__(self):
        self.front=None
        self.dequeSize=0
```

## 在 python 中的 deque 中插入元素

我们可以在队列的前后插入。为了在前面的队列中插入一个元素，我们将在链表的开头添加这个元素。由于链表的开始被 front 引用，我们将在当前被 front 引用的节点之前添加元素，然后将 dequeSize 增加 1。这可以如下实现。

```py
def insertAtFront(self,data):
        temp=Node(data)
        if self.front==None:
            self.front=temp
            self.dequeSize=self.dequeSize+1
        else:
            temp.next=self.front
            self.front=temp
            self.dequeSize=self.dequeSize+1
```

要在队列的后面插入一个元素，我们必须将该元素添加到链表的末尾。之后，我们将如下增加 dequeSize。

```py
 def insertAtRear(self,data):
        temp=Node(data)
        if self.front==None:
            self.front=temp
            self.dequeSize=self.dequeSize+1
        else:
            curr=self.front
            while curr.next!=None:
                curr=curr.next
            curr.next=temp
            self.dequeSize=self.dequeSize+1
```

## 在 python 中从队列中删除元素

我们可以从队列的前面和后面删除一个元素。在删除元素之前，我们将检查队列是否为空。如果 dequee 为空，即 front 指向 None，我们将使用 [python try except](https://www.pythonforbeginners.com/error-handling/python-try-and-except) 引发一个异常，并显示一条消息，说明 dequee 为空。

要从队列的前面删除一个元素，我们必须从前面引用的链表的开头删除这个元素。删除前面的元素后，我们将把 dequeSize 减 1。这可以如下实现。

```py
def delFromFront(self):
        try:
            if self.dequeSize==0:
                raise Exception("Deque is Empty")
            else:
                temp=self.front
                self.front=self.front.next
                self.dequeSize=self.dequeSize-1
                del temp
        except Exception as e:
            print(str(e))
```

为了删除队列后面的元素，我们将删除链表的最后一个元素。删除最后一个元素后，我们将把 dequeSize 减 1。这可以如下实现。

```py
def deleteFromRear(self):
        try:
            if self.dequeSize==0:
                raise Exception("Deque is Empty")
            else:
                curr=self.front
                prev=None
                while curr.next!=None:
                    prev=curr
                    curr=curr.next
                prev.next=curr.next
                self.dequeSize=self.dequeSize-1
                del curr
        except Exception as e:
            print(str(e))
```

## 检查队列的大小

要检查队列的大小，我们只需检查队列的 dequeSize 字段中的值。这可以如下进行。

```py
def dequeLength(self):
        return self.dequeSize
```

## 检查队列是否为空

要检查 deque 是否为空，我们必须检查 dequeSize 是否为零。我们可以实现 isEmpty()方法，如果 dequeSize 为零，该方法返回 True，否则返回 False，如下所示。

```py
def isEmpty(self):
        if self.dequeSize==0:
            return True
        return False
```

## 检查队列中的前后元素

为了检查队列的前端元素，我们将定义一个 getfront()方法，该方法返回链表的第一个元素，如下所示。

```py
def getfront(self):
        try:
            if self.dequeSize==0:
                raise Exception("Deque is Empty")
            else:
                return self.front.data
        except Exception as e:
            print(str(e))
```

为了检查队列的尾部元素，我们将定义一个方法 getrear()，该方法返回链表的最后一个元素，如下所示。

```py
def getrear(self):
        try:
            if self.dequeSize==0:
                raise Exception("Deque is Empty")
            else:
                curr=self.front
                while curr.next!=None:
                    curr=curr.next
                return curr.data
        except Exception as e:
            print(str(e))
```

python 中完整的工作代码实现 deque 如下。

```py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 23:41:28 2021

@author: aditya1117
"""
class Node:
    def __init__(self,data):
        self.data=data
        self.next=None

class Deque:
    def __init__(self):
        self.front=None
        self.dequeSize=0
    def isEmpty(self):
        if self.dequeSize==0:
            return True
        return False
    def dequeLength(self):
        return self.dequeSize
    def insertAtFront(self,data):
        temp=Node(data)
        if self.front==None:
            self.front=temp
            self.dequeSize=self.dequeSize+1
        else:
            temp.next=self.front
            self.front=temp
            self.dequeSize=self.dequeSize+1
    def insertAtRear(self,data):
        temp=Node(data)
        if self.front==None:
            self.front=temp
            self.dequeSize=self.dequeSize+1
        else:
            curr=self.front
            while curr.next!=None:
                curr=curr.next
            curr.next=temp
            self.dequeSize=self.dequeSize+1
    def delFromFront(self):
        try:
            if self.dequeSize==0:
                raise Exception("Deque is Empty")
            else:
                temp=self.front
                self.front=self.front.next
                self.dequeSize=self.dequeSize-1
                del temp
        except Exception as e:
            print(str(e))
    def deleteFromRear(self):
        try:
            if self.dequeSize==0:
                raise Exception("Deque is Empty")
            else:
                curr=self.front
                prev=None
                while curr.next!=None:
                    prev=curr
                    curr=curr.next
                prev.next=curr.next
                self.dequeSize=self.dequeSize-1
                del curr
        except Exception as e:
            print(str(e))
    def getfront(self):
        try:
            if self.dequeSize==0:
                raise Exception("Deque is Empty")
            else:
                return self.front.data
        except Exception as e:
            print(str(e))
    def getrear(self):
        try:
            if self.dequeSize==0:
                raise Exception("Deque is Empty")
            else:
                curr=self.front
                while curr.next!=None:
                    curr=curr.next
                return curr.data
        except Exception as e:
            print(str(e))
```

## 结论

在本文中，我们研究了 deque 背后的概念，并使用 python 中的链表实现了它。为了更深入地了解它，并理解 dequee 与内置数据结构(如 [python dictionary](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/) 、list 和 set)有何不同，请复制上面给出的完整代码，将其粘贴到您的 IDE 中，并试验 dequee 操作。请继续关注更多内容丰富的文章。