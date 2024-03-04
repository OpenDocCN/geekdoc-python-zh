# Python 中的堆栈

> 原文：<https://www.pythonforbeginners.com/data-types/stack-in-python>

在 python 中，有内置的数据结构，如列表、元组、集合和字典，但我们可能需要 python 程序中的一些附加功能。例如，如果我们的程序需要后进先出(LIFO)功能，那么没有内置的数据结构提供这种功能。我们可以使用堆栈来实现 python 中的后进先出功能。在本文中，我们将研究堆栈数据结构背后的概念，并用 python 实现它。

## 什么是堆栈？

堆栈是一种线性数据结构，在这种结构中，我们只能访问或删除最后添加到堆栈中的元素。在现实生活中，我们可以以一叠盘子为例，我们只能取出最上面的盘子，因为它是最后添加的。我们可以在 python 中对堆栈数据结构执行以下操作。

1.  插入元素(推动元素)
2.  移除元素(弹出元素)
3.  检查堆栈是否为空
4.  检查堆栈中最顶端的元素
5.  检查堆栈的大小

在接下来的小节中，我们将实现 stack 和它的所有部分，并将在 python 中实现它们。

## 如何在 python 中实现 stack？

在本教程中，我们将定义一个名为 Stack 的类，并使用 python 列表实现该堆栈。在 Stack 类中，我们将有一个包含添加到堆栈中的数据的列表和一个存储堆栈大小的变量。所有操作，如 push、pop、检查堆栈的大小、检查堆栈的最顶端元素以及检查堆栈是否为空，都将在恒定时间内执行，因此时间复杂度为 O(1)。堆栈类将被定义如下。stackList 被初始化为空列表，stackSize 被初始化为 0。

```py
class Stack:
    def __init__(self):
        self.stackList=[]
        self.stackSize=0
```

## 在 python 中将项目推入堆栈

要将一个项目插入堆栈，也就是将一个元素推入堆栈，我们只需将该元素添加到列表中，然后将 stackSize 变量递增 1。为了实现该操作，我们定义了一个方法，该方法将元素作为参数，并将元素添加到堆栈中的列表中。然后它递增 stackSize 变量。下面是 push()方法的实现。

```py
def push(self,item):
        self.stackList.append(item)
        self.stackSize+=1
```

## 用 python 从堆栈中弹出项目

要从堆栈中移除一个项目，即从堆栈中弹出一个元素，我们必须从堆栈中移除最后添加到它的元素。当我们在推操作时将元素添加到堆栈中的列表时，列表中的最后一个元素将是最近的元素，并将从堆栈中移除。所以我们只需从列表中删除最后一个元素。

为了实现 pop 操作，我们将实现 pop()方法，该方法首先检查堆栈中元素的数量是否大于 0。如果是，那么它将从列表中删除最后一个元素，并将 stackSize 减 1。如果堆栈中的元素数为 0，它将显示一条错误消息，说明列表为空。对于这个任务，我们将使用异常处理，使用 [python try except](https://www.pythonforbeginners.com/error-handling/python-try-and-except) 在堆栈大小为 0 时引发异常。pop()方法可以按如下方式实现。

```py
def pop(self):
        try:
            if self.stackSize==0:
                raise Exception("Stack is Empty, returning None")
            temp=self.stackList.pop()
            self.stackSize-=1
            return temp
        except Exception as e:
            print(str(e))
```

## 检查堆栈的大小

要检查堆栈的大小，我们只需检查 stackSize 变量的值。对于这个操作，我们将实现 size()方法，该方法返回 stackSize 变量的值，如下所示。

```py
def size(self):
        return self.stackSize
```

## 检查堆栈是否为空

要检查堆栈是否没有元素，即它是否为空，我们必须检查 stackSize 变量是否为 0。对于这个操作，我们将实现 isEmpty()方法，如果 stackSize 变量为 0，则返回 True，否则返回 false。下面是 isEmpty()方法的实现。

```py
def isEmpty(self):
        if self.stackSize==0:
            return True
        else:
            return False
```

## 检查堆栈的最顶层元素

堆栈最顶端的元素将是最近添加到其中的元素。要检查栈顶的元素，我们只需返回栈中列表的最后一个元素。对于这个操作，我们将实现 top()方法，它首先检查堆栈是否为空，即 stackSize 是否为 0，如果是，它将打印一条消息，说明堆栈为空。否则它将返回列表的最后一个元素。这可以如下实现。

```py
def top(self):
        try:
            if self.stackSize==0:
                raise Exception("Stack is Empty, returning None")
            return self.stackList[-1]
        except Exception as e:
            print(str(e))
```

下面是用 python 实现 stack 的完整工作代码。

```py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 19:38:55 2021

@author: aditya1117
"""

class Stack:
    def __init__(self):
        self.stackList=[]
        self.stackSize=0
    def push(self,item):
        self.stackList.append(item)
        self.stackSize+=1
    def pop(self):
        try:
            if self.stackSize==0:
                raise Exception("Stack is Empty, returning None")
            temp=self.stackList.pop()
            self.stackSize-=1
            return temp
        except Exception as e:
            print(str(e))
    def size(self):
        return self.stackSize
    def isEmpty(self):
        if self.stackSize==0:
            return True
        else:
            return False
    def top(self):
        try:
            if self.stackSize==0:
                raise Exception("Stack is Empty, returning None")
            return self.stackList[-1]
        except Exception as e:
            print(str(e))

#Execution
s=Stack()
#push element
s.push(1)
#push element
s.push(2)
#push element
s.push(3)
print("popped element is:")
print(s.pop())
#push an element
s.push(4)
print("topmost element is:")
print(s.top())
```

输出:

```py
popped element is:
3
topmost element is:
4
```

## 结论

在本文中，我们研究并实现了 python 中的栈数据结构。我们已经看到了堆栈的操作，您可能会发现它与 [python dictionary](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/) ，list，set e.t.c .非常不同。堆栈广泛用于现实世界的应用程序中，如表达式处理、回溯、函数调用和许多其他操作，您应该对此有所了解。请继续关注更多内容丰富的文章。