# Python min()方法

> 原文：<https://www.askpython.com/python/built-in-methods/python-min-method>

## 介绍

在本教程中，我们将了解 **Python min()方法**的用法。

基本上，Python `min()`方法返回一组被传递的值或被传递的 iterable 的元素中的最小值。

## 了解 Python min()方法

下面给出了在 Python 中使用`min()`方法的一般语法。使用这个我们可以在一个可迭代的元素([列表](https://www.askpython.com/python/list/python-list)、[元组](https://www.askpython.com/python/tuple/python-tuple)、[字符串](https://www.askpython.com/python/string)等)中找到最小值。).

```py
min(iterable, *[, key, default])

```

为了找到一组项目中的最小值，我们可以直接将它们传递给用逗号分隔的`min()`函数(“**，**”)。

```py
min(arg1, arg2, *args[, key])

```

这里，

*   **iterable** 包含需要寻找最小值的值，
*   **键**是单行订购功能，
*   **default** 是函数返回的默认值，如果传递的 iterable 为空，
*   **arg1，arg2，… argn** 是 min()函数将返回最小值的一组值。

现在我们已经理解了使用`min()`方法的语法，让我们看一些例子来更好地理解它的工作原理。

## 使用 Python min()方法

如前所述，我们可以使用 Python `min()`函数在作为参数传递的一组值中或者在传递的 iterable 的元素中找到最小值。

现在，为了理解工作原理，我们举几个例子。

### 1.使用可迭代对象

`min()`函数广泛用于查找出现在**可迭代**中的最小值，如列表、元组、列表列表、元组列表等。在简单列表和元组的情况下，它返回 iterable 中存在的最小值。

看看下面给出的例子。

```py
# initialisation of list
list1 = [23,45,67,89]

# finding min element
print("Min value is : ", min(list1, default=0))

```

**输出**:

```py
Min value is : 23

```

在这里，将列表 **list1** 直接传递给`min()`方法会给出列表中所有元素的最小值，即 **23** 。将`default`值设置为 **0** ，这样如果传递的 iterable 为空，该方法将返回默认值 **(0)** 。

对于字符的**列表，`min()`方法返回具有最小 ASCII 值的元素。**

### 2.有多个参数

当我们向`min()`方法传递多个参数时，它返回所有参数中最小的一个。

**注意**，我们可以向`min()`方法传递多个值以及多个可迭代对象。对于多个 iterabless，该方法返回第一个元素最小的 iterable(值在第**个**索引处)。

下面的例子很容易解释这一点:

```py
# initialisation of lists
list1 = [23,45,67]
list2 = [89,65,34]
list3 = [19,90,31]

# finding min element
print("Min among set of values is : ", min(765,876,434))
print("Min list among the given lists is : ", min(list1,list2,list3))

```

**输出**:

```py
Min among set of values is :  434
Min list among the given lists is :  [19, 90, 31]

```

对于上面的例子，当我们将多个值作为参数传递给`min()`方法时，它只是返回给我们最小值( **434**

然而，对于 list1、list2 和 list3，它返回 **list3** ，因为它具有最小的**第 0**索引值( **19** )。

### 3.带按键功能

正如我们前面提到的，**键**函数是一个单行排序函数，它根据哪个参数来确定要返回的最小值。

让我们举个例子来理解这个关键概念。

```py
# initialisation of variables
list_of_tuples = [(9, 2, 7), (6, 8, 4), (3, 5, 1)]

list1 = [23,45]
list2 = [89,65,34]
list3 = [19,90,31,67]

def ret_2nd_ele(tuple_1):
    return tuple_1[1]

#find Min from a list of tuples with key on the basis of the 2nd element
print("Min in list of tuples : ", min(list_of_tuples, key=ret_2nd_ele))

#find min from a bunch of lists on the basis of their length
print("List with min length : ", min(list1,list2,list3,key=len))

```

**输出**:

```py
Min in list of tuples :  (9, 2, 7)
List with min length :  [23, 45]

```

*   我们首先初始化元组列表以及其他三个不同长度的整数列表，
*   然后我们定义一个函数`ret_2nd_ele()`,它返回传递的元组的第 2 个元素或第 1 个索引项，
*   之后，我们将 **list_of_tuples** 传递给以`ret_2nd_ele()`函数为键的`min()`方法，
*   我们再次将三个列表**列表 1** 、**列表 2** 和**列表 3** 作为参数传递给`min()`方法，并将一个键设置为内置的`len()`方法。

这样，我们得到了元组列表中具有最小第 2 个元素(第 1 项)的元组。以及三个列表中长度最小的列表(使用`len()`)，即`list1`。

## 结论

请记住，传递一个没有为`min()`方法设置默认值的**空** iterable 会引发一个`ValueError`。

这就是关于 Python 中的`min()`方法的教程。如有任何进一步的问题，欢迎使用下面的评论。

## 参考

*   python min()–journal dev Post
*   [min()](https://docs.python.org/3/library/functions.html#min)–Python 文档
*   [python min 函数如何工作](https://stackoverflow.com/questions/14976031/how-does-the-python-min-function-works)–堆栈溢出问题