# Python 中的链表

> 原文：<https://www.pythonforbeginners.com/lists/linked-list-in-python>

链表是一种数据结构，它包含由链接连接的数据对象。每个链表由具有数据字段和对链表中下一个节点的引用的节点组成。在本文中，我们将研究链表背后的基本概念，并用 python 实现它。

## 链表中的节点是什么？

节点是一个对象，它有一个数据字段和一个指向链表中另一个节点的指针。下图显示了一个简单的节点结构。

![](img/e4a0afd4504101ca7137629c0dd23d2c.png)



Node of a linked list in python

我们可以使用具有两个字段(即 data 和 next)的类节点来实现上述对象，如下所示。

```py
class Node:
    def __init__(self,data):
        self.data=data
        self.next=None
```

链表由许多这样的节点组成。链表中有一个头指针指向链表中的第一个节点，或者当链表为空时没有头指针。下图描述了一个有三个节点的链表。

![](img/6f943220bd6e670521972d6281dd5b3e.png)



Linked list in python

我们可以看到最后一个节点的 next 字段指向 None，引用头指向第一个节点。空链表将是一个头指针指向 None 的链表。可以用 python 创建一个空链表，如下所示。

```py
class linkedList:
    def __init__(self):
        self.head=None
```

## 向链表中插入元素

我们可以在链表中的第一个位置、最后一个位置或者两者之间的任何位置插入一个元素。

为了在链表的开头插入一个元素，我们将首先用给定的数据创建一个节点，并将它的下一个引用分配给第一个节点，也就是头指向的地方。然后，我们将 head 引用指向新节点。为了执行这个操作，我们如下实现方法 insertAtBeginning。

```py
def insertAtBeginning(self,data):
        temp=Node(data)
        if self.head==None:
            self.head=temp
        else:
            temp.next=self.head
            self.head=temp
```

要在链表的末尾插入一个元素，我们只需要找到下一个元素没有引用的节点，即最后一个节点。然后，我们用给定的数据创建一个新节点，并将最后一个节点的下一个元素指向新创建的节点。为了执行这个操作，我们如下实现方法 insertAtEnd。

```py
def insertAtEnd(self,data):
        temp=Node(data)
        if self.head==None:
            self.head=temp
        else:
            curr=self.head
            while curr.next!=None:
                curr=curr.next
            curr.next=temp
```

要在任何其他给定位置插入元素，我们可以计算节点数，直到到达该位置，然后将新节点的下一个元素指向当前节点的下一个节点，并将当前节点的下一个引用指向新节点。这可以使用 insertAtGivenPosition 方法实现，如下所示。

```py
def insertAtGivenPosition(self,data,position):
        count=1
        curr=self.head
        while count<position-1 and curr!=None:
            curr=curr.next
            count+=1
        temp=Node(data)
        temp.next=curr.next
        curr.next=temp
```

## 遍历链表

为了遍历 python 中的链表，我们将从头开始，打印数据并移动到下一个节点，直到到达 None，即链表的末尾。下面的 traverse()方法实现了在 python 中遍历链表的程序。

```py
def traverse(self):
        curr=self.head
        while curr!=None:
            print(curr.data)
            curr=curr.next
```

## 删除节点

我们可以从链表的开始或者结束或者两个节点之间删除一个节点。

要删除链表的第一个节点，我们将首先检查链表的头是否指向 None，如果是，那么我们将使用 [python try except](https://www.pythonforbeginners.com/error-handling/python-try-and-except) 抛出一个异常，并显示一条消息，说明链表为空。否则，我们将删除 head 引用的当前节点，并将 head 指针移动到下一个节点。这可以如下实现。

```py
def delFromBeginning(self):
        try:
            if self.head==None:
                raise Exception("Empty Linked List")
            else:
                temp=self.head
                self.head=self.head.next
                del temp
        except Exception as e:
            print(str(e))
```

为了删除链表的最后一个节点，我们将遍历链表中的每个节点，并检查当前节点的下一个节点的 next 指针是否指向 None，如果是，则当前节点的下一个节点是最后一个节点，它将被删除。这可以如下实现。

```py
def delFromEnd(self):
        try:
            if self.head==None:
                raise Exception("Empty Linked List")
            else:
                curr=self.head
                prev=None
                while curr.next!=None:
                    prev=curr
                    curr=curr.next
                prev.next=curr.next
                del curr
        except Exception as e:
            print(str(e))
```

要删除链表之间的节点，在每一个节点，我们将检查下一个节点的位置是否是要删除的节点，如果是，我们将删除下一个节点，并将下一个引用分配给要删除的节点的下一个节点。这可以如下进行。

```py
def delAtPos(self,position):
        try:
            if self.head==None:
                raise Exception("Empty Linked List")
            else:
                curr=self.head
                prev=None
                count=1
                while curr!=None and count<position:
                    prev=curr
                    curr=curr.next
                    count+=1
                prev.next=curr.next
                del curr
        except Exception as e:
            print(str(e))
```

下面是用 python 实现链表的 python 代码。

```py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 18:28:15 2021

@author: aditya1117
"""

class Node:
    def __init__(self,data):
        self.data=data
        self.next=None
class linkedList:
    def __init__(self):
        self.head=None
    def insertAtBeginning(self,data):
        temp=Node(data)
        if self.head==None:
            self.head=temp
        else:
            temp.next=self.head
            self.head=temp
    def insertAtEnd(self,data):
        temp=Node(data)
        if self.head==None:
            self.head=temp
        else:
            curr=self.head
            while curr.next!=None:
                curr=curr.next
            curr.next=temp
    def insertAtGivenPosition(self,data,position):
        count=1
        curr=self.head
        while count<position-1 and curr!=None:
            curr=curr.next
            count+=1
        temp=Node(data)
        temp.next=curr.next
        curr.next=temp
    def traverse(self):
        curr=self.head
        while curr!=None:
            print(curr.data)
            curr=curr.next
    def delFromBeginning(self):
        try:
            if self.head==None:
                raise Exception("Empty Linked List")
            else:
                temp=self.head
                self.head=self.head.next
                del temp
        except Exception as e:
            print(str(e))
    def delFromEnd(self):
        try:
            if self.head==None:
                raise Exception("Empty Linked List")
            else:
                curr=self.head
                prev=None
                while curr.next!=None:
                    prev=curr
                    curr=curr.next
                prev.next=curr.next
                del curr
        except Exception as e:
            print(str(e))
    def delAtPos(self,position):
        try:
            if self.head==None:
                raise Exception("Empty Linked List")
            else:
                curr=self.head
                prev=None
                count=1
                while curr!=None and count<position:
                    prev=curr
                    curr=curr.next
                    count+=1
                prev.next=curr.next
                del curr
        except Exception as e:
            print(str(e)) 
```

## 结论

在这篇文章中，我们研究了链表并在 python 中实现了它。您可能已经观察到链表与内置数据结构(如 [python dictionary](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/) 、list 和 set e.t.c)的不同之处。请复制工作代码并对其进行实验，以获得更多见解。请继续关注更多内容丰富的文章。