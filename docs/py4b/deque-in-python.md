# python

> 原文：<https://www.pythonforbeginners.com/deque/deque-in-python>

双端队列是线性数据结构，我们可以用它来执行后进先出(LIFO)操作和先进先出(FIFO)操作。Deques 在现实生活中有很多应用，比如在软件中实现撤销操作，在 web 浏览器中存储浏览历史。在本文中，我们将研究 deque 背后的基本概念，并用 python 实现它们。

## 如何用 python 实现 deque？

如上所述，deques 是线性数据结构。因此，我们可以使用 python 中的 list 实现 deque。要使用 list 实现 deque，我们必须在 list 的两边插入和删除元素。我们还可以执行像检查队列长度或检查队列是否为空这样的操作。对于这个任务，我们将使用一个 dequeSize 方法来记录 deque 中元素的数量。可以使用下面的类定义在 python 中实现 dequeue，其中我们定义了一个名为 dequeList 的空列表来初始化空 dequeue，并将 dequeSize 初始化为 0，如下所示。

```py
class Deque:
    def __init__(self):
        self.dequeList=list()
        self.dequeSize=0
```

## python 中 deque 中的插入

我们可以在队列的前面和后面插入一个元素。要在 deque 的前面插入一个元素，我们只需使用 insert()方法在 dequeList 的开头插入该元素。然后我们将把 dequeSize 增加 1。这可以如下进行。

```py
def insertAtFront(self,data):
        self.dequeList.insert(0,data)
        self.dequeSize=self.dequeSize+1
```

要在 deque 的后面插入一个元素，我们将使用 append()方法将该元素附加到 dequeList 的末尾，然后将 dequeSize 增加 1。这可以如下进行。

```py
def insertAtRear(self,data):
        self.dequeList.append(data)
        self.dequeSize=self.dequeSize+1
```

## 在 python 中从队列中删除元素

我们可以从队列的前面和后面删除一个元素。在删除一个元素之前，我们将首先检查队列是否为空，并将使用 [python try except](https://www.pythonforbeginners.com/error-handling/python-try-and-except) 抛出一个异常，即队列为空。否则，我们将继续从队列中删除该元素。

要删除 dequeue 前面的元素，我们将使用 pop()方法删除 dequeList 的第一个元素，然后将 dequeSize 减 1。这可以如下进行。

```py
def deleteFromFront(self):
        try:
            if self.dequeSize==0:
                raise Exception("Deque is Empty")
            else:
                self.dequeList.pop(0)
                self.dequeSize=self.dequeSize-1
        except Exception as e:
            print(str(e))
```

为了从 deque 的后面删除一个元素，我们将删除 dequeList 的最后一个位置的元素，并将 dequeSize 减 1。这可以如下进行。

```py
def deleteFromRear(self):
        try:
            if self.dequeSize==0:
                raise Exception("Deque is Empty")
            else:
                self.dequeList.pop(-1)
                self.dequeSize=self.dequeSize-1
        except Exception as e:
            print(str(e))
```

## 检查队列的长度

要检查 dequee 的长度，我们只需检查 dequee 的 dequeSize 字段中的值，该字段保存了 dequee 的长度。我们可以实现 length()方法来检查 python 中 deque 的长度，如下所示。

```py
def dequeLength(self):
        return self.dequeSize
```

## 检查队列是否为空

要检查 deque 是否为空，我们只需检查 dequeSize 字段中是否有值 0。我们可以用 python 实现一个方法 isEmpty()来检查队列是否为空，如下所示。

```py
def isEmpty(self):
        if self.dequeSize==0:
            return True
        return False
```

## 检查前后元件

为了检查 deque 的 front 元素，我们可以如下实现 front()方法。

```py
def front(self):
        try:
            if self.dequeSize==0:
                raise Exception("Deque is Empty")
            else:
                return self.dequeList[0]
        except Exception as e:
            print(str(e))
```

为了检查 deque 的尾部元素，我们可以如下实现 rear()方法。

```py
def rear(self):
        try:
            if self.dequeSize==0:
                raise Exception("Deque is Empty")
            else:
                return self.dequeList[-1]
        except Exception as e:
            print(str(e))
```

下面是 python 中使用 list 的 deque 的完整实现。

```py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 18:31:28 2021

@author: aditya1117
"""

class Deque:
    def __init__(self):
        self.dequeList=list()
        self.dequeSize=0
    def isEmpty(self):
        if self.dequeSize==0:
            return True
        return False
    def dequeLength(self):
        return self.dequeSize
    def insertAtFront(self,data):
        self.dequeList.insert(0,data)
        self.dequeSize=self.dequeSize+1
    def insertAtRear(self,data):
        self.dequeList.append(data)
        self.dequeSize=self.dequeSize+1
    def deleteFromFront(self):
        try:
            if self.dequeSize==0:
                raise Exception("Deque is Empty")
            else:
                self.dequeList.pop(0)
                self.dequeSize=self.dequeSize-1
        except Exception as e:
            print(str(e))
    def deleteFromRear(self):
        try:
            if self.dequeSize==0:
                raise Exception("Deque is Empty")
            else:
                self.dequeList.pop(-1)
                self.dequeSize=self.dequeSize-1
        except Exception as e:
            print(str(e))
    def front(self):
        try:
            if self.dequeSize==0:
                raise Exception("Deque is Empty")
            else:
                return self.dequeList[0]
        except Exception as e:
            print(str(e))
    def rear(self):
        try:
            if self.dequeSize==0:
                raise Exception("Deque is Empty")
            else:
                return self.dequeList[-1]
        except Exception as e:
            print(str(e))
```

## 结论

在本文中，我们研究了 deque 背后的概念，并使用 python 中的 list 实现了它。为了更深入地了解它，并理解 dequee 与内置数据结构(如 [python dictionary](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/) 、list 和 set)有何不同，复制上面示例中给出的完整代码，将其粘贴到您的 IDE 中，并试验 dequee 操作。请继续关注更多内容丰富的文章。