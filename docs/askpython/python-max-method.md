# Python max()方法

> 原文：<https://www.askpython.com/python/built-in-methods/python-max-method>

## 介绍

在本教程中，我们将了解如何使用**Python max()方法**。基本上，Python `max()`方法返回一组被传递的值或被传递的 iterable 的元素中的最大值。

## 使用 Python max()方法

下面是使用 Python `max()`方法在**可迭代**中寻找最大值的语法，

```py
max(iterable, *[, key, default])

```

这里，

*   **iterable** 是包含最大值的对象，
*   **键**指定单参数排序函数，
*   而 **default** 是方法在传递的 iterable 为空的情况下返回的默认值。

为了找到作为参数传递的两个或多个值中的最大值，

```py
max(arg1, arg2, *args[, key]) 

```

这里，

*   **arg1，arg2** ，…。 **argn** 是 **n** 个值，其中`max()`方法将返回最大值，
*   关键还是排序功能。

## 使用 Python max()方法

我们可以以各种方式使用`max()`方法来寻找给定 iterable 或两个或更多参数的最大值。

让我们看看这个方法是如何处理一个 iterable 对象，两个或更多的值，指定的键函数，以及作为参数传递的多个 iterable 对象。

## 使用可迭代对象

对于下面的例子，我们考虑一个**列表**,其中有一些值，我们要找到最大的元素。仔细看下面给出的代码。

```py
#initialisation of list
list1 = [ 1,3,4,7,0,4,8,2 ]

#finding max element
print("max value is : ", max(list1,default=0))

```

**输出**:

```py
max value is :  8

```

正如我们所看到的，对于上面的代码，我们初始化了一个列表，`list1`并直接将其传递给`max()`方法，默认值设置为 **0** 。该函数返回 **8** ，因为它是最大值。

如果列表**为空**，该函数将传递默认值，即 **0** 。

## 向 Python max()方法传递两个或多个值

当两个或更多的值被传递给`max()`方法时，它返回所有值中的最大值。这些参数可以是整数、浮点值、字符甚至字符串。

让我们举一个例子，

```py
print("max value is : ", max(6,1,73,6,38))

```

**输出**:

```py
max value is :  73

```

如我们所愿，我们得到了最大值， **73。**

## 带按键功能

正如我们[前面提到的](https://www.askpython.com/python/list/python-sort-list)，这个键是一个单行排序函数，在这个函数的基础上可以找到一组值中的最大值。

例如，如果我们想从元组列表的**中找到一个元组，该元组具有第二个**元素**的最大值。让我们看看如何做到这一点。**

```py
#initialisation of variables
list1 = [(9,2,7), (6,8,4), (3,5,1)]

def f(tuple_1):
    return tuple_1[1]

print("max : ", max(list1, key=f))

```

**输出**:

```py
max :  (6, 8, 4)

```

这里，`f()`是一个用户自定义函数，返回传递的元组的第**第 2 个**元素。将这个函数作为一个**键**传递给`max()`方法可以确保返回一个具有最大的第二个元素的元组。对于我们的例子，它是 **( 6，8，4)。**

## 将多个可重复项作为参数传递

如前所述，Python max()方法也可以返回多个可迭代元素中最大的一个作为参数。这些参数可以是可迭代的，如字符串、字符、元组、列表等..

默认为**，`max()`方法返回列表、元组等最大**第 0th】元素的对象。对于字符串，它比较传递的每个字符串的第一个字符。****

**下面我们以元组为例。仔细看代码。**

```py
#initialisation of variables
tuple1 = (5,23,7)
tuple2 = (4,1,7)
tuple3 = (7,37,1)

print("max : ", max(tuple1,tuple2,tuple3)) 
```

**输出:**

```py
max :  (7, 37, 1) 
```

**在这个例子中，**个带有一些初始值的**元组被直接传递给了`max()`方法。其返回具有最大第一元素的元组，即 **(7，37，1)** 。**

## **结论**

**所以在本教程中，我们学习了 Python `max()`方法，它的用法和工作原理。**

**请记住，如果没有设置**默认的**值，并且将一个空的 iterable 作为参数传递给`max()`函数，就会产生一个**值错误**。**

**关于这个话题的任何进一步的问题，请随意使用下面的评论。**

## **参考**

*   **python max()–日志开发帖子，**
*   **[max()](https://docs.python.org/3.7/library/functions.html#max)–Python 文档，**
*   **[使用‘key’和 lambda 表达式的 python max 函数](https://stackoverflow.com/questions/18296755/python-max-function-using-key-and-lambda-expression)–堆栈溢出问题。**