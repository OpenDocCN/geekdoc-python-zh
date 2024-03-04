# Python 中的队列

> 原文：<https://www.pythonforbeginners.com/queue/queue-in-python>

你一定在现实生活中见过排队等候预约医生或在餐馆点餐的情形。队列数据结构遵循后进先出(LIFO)顺序来访问元素。首先添加的元素只能被访问或删除。在本文中，我们将研究队列数据结构背后的基本概念，并用 python 实现它。

## 如何用 python 实现队列？

队列是一种线性数据结构，在这种结构中，我们只能访问或删除最先添加到队列中的元素。我们将使用一个列表实现一个队列。为了实现，我们将定义一个队列类，它将有一个包含元素的列表和一个包含列表长度的 queueLength 字段。python 中的队列类实现如下。

```py
class Queue:
    def __init__(self):
        self.queueList=list()
        self.queueLength=0
```

## 在 python 中向队列添加元素

当我们向队列中添加一个元素时，这个操作被称为入队操作。要实现入队操作，我们只需将元素添加到队列的列表中。然后，我们将队列长度增加 1。实现入队操作的 enQueue()方法将元素作为参数并执行操作。这可以如下实现。

```py
def enQueue(self,data):
        self.queueList.append(data)
        self.queueLength=self.queueLength+1
```

## 在 python 中从队列中移除元素

当我们从队列中删除一个元素时，这个操作称为出列操作。为了实现出列操作，我们将只弹出队列中列表的第一个元素。然后，我们将队列长度减 1。在出队操作之前，我们将检查队列是否为空。如果是，将使用 [python try except](https://www.pythonforbeginners.com/error-handling/python-try-and-except) 引发一个异常，并显示队列为空的消息。否则将执行出列操作。实现出列操作的 deQueue()方法可以按如下方式实现。

```py
def deQueue(self):
        try:
            if self.queueLength==0:
                raise Exception("Queue is Empty")
            else:
                temp=self.queueList.pop(0)
                self.queueLength=self.queueLength-1
                return temp
        except Exception as e:
            print(str(e))
```

## 找出队列的长度

要找到队列的长度，我们只需查看 queueLength 变量的值。length()方法实现如下。

```py
def length(self):
        return self.queueLength
```

## 检查队列是否为空

要检查队列是否为空，我们必须确定 queueLength 是否为 0。isEmpty()方法将实现如下逻辑。

```py
def isEmpty(self):
        if self.queueLength==0:
            return True
        else:
            return False
```

## 获取队列中的前一个元素

要获得队列中的前一个元素，我们必须返回队列中列表的第一个元素。这可以如下实现。

```py
def front(self):
        try:
            if self.queueLength==0:
                raise Exception("Queue is Empty")
            else:
                temp=self.queueList[-1]
                return temp
        except Exception as e:
            print(str(e))
```

下面是用 python 实现队列的完整代码。

```py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 20:15:56 2021

@author: aditya1117
"""

class Queue:
    def __init__(self):
        self.queueList=list()
        self.queueLength=0
    def enQueue(self,data):
        self.queueList.append(data)
        self.queueLength=self.queueLength+1
    def deQueue(self):
        try:
            if self.queueLength==0:
                raise Exception("Queue is Empty")
            else:
                temp=self.queueList.pop(0)
                self.queueLength=self.queueLength-1
                return temp
        except Exception as e:
            print(str(e))
    def isEmpty(self):
        if self.queueLength==0:
            return True
        else:
            return False
    def length(self):
        return self.queueLength
    def front(self):
        try:
            if self.queueLength==0:
                raise Exception("Queue is Empty")
            else:
                temp=self.queueList[-1]
                return temp
        except Exception as e:
            print(str(e))
```

## 结论

在本文中，我们已经理解了队列背后的概念，并用 python 实现了它。复制上面给出的完整代码，将其粘贴到您的 IDE 中，尝试操作以理解概念，并查看队列与其他数据结构(如 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)、列表和集合)有何不同。请继续关注更多内容丰富的文章。