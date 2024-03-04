# 在 Python 中实现堆栈

> 原文：<https://www.pythonforbeginners.com/data-types/implement-stack-in-python>

栈是一种数据结构，它遵循后进先出(LIFO)的顺序来访问元素。在堆栈中，我们只能访问最近添加的元素。堆栈在表达式处理、回溯和函数调用等应用中有很多用途。在本文中，我们将尝试用 python 实现带链表的堆栈数据结构。

## 使用链表在 Python 中实现堆栈

链表是一种线性数据结构，其中每个数据对象指向另一个对象。为了用链表实现一个堆栈，我们将不得不执行插入和删除，以及从链表中删除元素，这样它将花费恒定的时间。只有当我们在链表的开头插入和删除元素时，这才是可能的。

为了在 python 中实现带链表的堆栈，我们将首先定义一个 node 对象，它将包含当前元素，并使用引用 next 指向插入它之前的节点。在 python 中，该节点可以按如下方式实现。

```py
class Node:
    def __init__(self,data,size):
        self.data=data
        self.next=None
```

当我们用链表实现一个堆栈时，它将有一个名为 top 的元素，这个元素将引用链表中最近插入的元素。所有其他元素都可以通过它来访问。我们可以用 python 实现一个空栈，top 初始化为 None，stackSize 初始化为 0，如下所示。

```py
class Stack:
    def __init__(self):
        self.top=None
        self.stackSize=0
```

## 在 Python 中实现栈中的推送操作

当我们将一个元素插入到堆栈中时，这个操作叫做 push 操作。为了在链表的帮助下实现栈中的 push 操作，对于每一次插入，我们将在链表的开头添加要插入的元素。这样，最近的元素将总是在链表的开始。然后，我们将堆栈的顶部元素指向链表的开头，并将 stackSize 递增 1。可以使用 python 中的链表实现推送操作，如下所示。

```py
 def push(self,data):
        temp=Node(data)
        if self.top is None:
            self.top=temp
            self.stackSize= self.stackSize+1
        else:
            temp.next=self.top
            self.top=temp
            self.stackSize=self.stackSize+1
```

## 用 Python 实现堆栈中的弹出操作

当我们从堆栈中取出一个元素时，这个操作被称为弹出操作。要从堆栈中取出的元素是最近添加到堆栈中的元素。正如我们所知，在推送操作中，我们将最近的元素添加到堆栈顶部指向的链表的开头，因此我们将删除顶部指向的元素，并将顶部指向下一个最近的元素。在执行弹出操作之前，我们将检查堆栈是否为空。如果堆栈为空，即 top 指向 None，我们将使用 [python try except](https://www.pythonforbeginners.com/error-handling/python-try-and-except) 引发一个异常，并显示堆栈为空的消息。我们可以如下实现 pop 操作。

```py
 def pop(self):
        try:
            if self.top == None:
                raise Exception("Stack is Empty")
            else:
                temp=self.top
                self.top=self.top.next
                tempdata=temp.data
                self.stackSize= self.stackSize-1
                del temp
                return tempdata
        except Exception as e:
            print(str(e))
```

## 如何访问栈顶元素

由于栈顶的元素被 top 引用，我们只需返回 top 指向的节点中的数据。它可以按如下方式实现。

```py
def top_element(self):
        try:
            if self.top == None:
                raise Exception("Stack is Empty")
            else:
                return self.top.data
        except Exception as e:
            print(str(e))
```

## 获取堆栈的大小

因为我们在 stackSize 变量中保留了当前的堆栈大小，所以我们只需返回 stackSize 变量 top 中的值。这可以如下进行。

```py
def size(self):
        return self.stackSize
```

## 检查堆栈是否为空

元素 top 指的是堆栈中最近的元素。如果堆栈中没有元素，top 将指向 None。因此，要检查堆栈是否为空，我们只需检查 top 是否指向 None 或者 stackSize 变量的值是否为 0。我们可以实现 isEmpty()方法来检查堆栈是否为空，如下所示。

```py
def isEmpty(self):
        if self.stackSize==0:
            return True
        else:
            return False
```

使用 python 中的链表实现堆栈的完整工作代码如下。

```py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 00:28:19 2021

@author: aditya1117
"""

class Node:
    def __init__(self,data):
        self.data=data
        self.next=None
class Stack:
    def __init__(self):
        self.top=None
        self.stackSize=0
    def push(self,data):
        temp=Node(data)
        if self.top is None:
            self.top=temp
            self.stackSize= self.stackSize+1
        else:
            temp.next=self.top
            self.top=temp
            self.stackSize=self.stackSize+1
    def pop(self):
        try:
            if self.top == None:
                raise Exception("Stack is Empty")
            else:
                temp=self.top
                self.top=self.top.next
                tempdata=temp.data
                self.stackSize= self.stackSize-1
                del temp
                return tempdata
        except Exception as e:
            print(str(e))
    def isEmpty(self):
        if self.stackSize==0:
            return True
        else:
            return False
    def size(self):
        return self.stackSize
    def top_element(self):
        try:
            if self.top == None:
                raise Exception("Stack is Empty")
            else:
                return self.top.data
        except Exception as e:
            print(str(e))
s=Stack()
s.push(1)
print(s.size())

s.push(2)
print(s.size())

print(s.pop())
print(s.size())
print(s.pop())
print(s.stackSize)

print(s.top_element())
print(s.isEmpty())
```

输出:

```py
1
2
2
1
1
0
Stack is Empty
None
True
```

## 结论

在本文中，我们使用 python 中的链表实现了 stack 及其所有操作。为了更深入地了解它，并理解 stack 与内置数据结构(如 [python dictionary](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/) 、list 和 set)有何不同，复制上面示例中给出的完整代码，并试验其中的操作。请继续关注更多内容丰富的文章。