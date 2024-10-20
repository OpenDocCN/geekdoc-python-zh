# Python 复制–执行深层和浅层复制

> 原文：<https://www.askpython.com/python-modules/python-copy>

在本文中，我们将研究如何使用 Python *Copy* 模块来执行深度和浅度复制操作。

那么，我们所说的深层拷贝和浅层拷贝是什么意思呢？

让我们看一看，用说明性的例子！

* * *

## 为什么我们需要 Python 复制模块？

在 Python 中，一切都用对象来表示。因此，在很多情况下，我们可能需要直接复制对象。

在这些情况下，我们不能直接使用赋值操作符。

赋值背后的要点是多个变量可以指向同一个对象。这意味着，如果对象使用这些变量中的任何一个发生变化，变化将会在任何地方反映出来！

下面的例子说明了这个问题，使用了一个共享的[列表对象](https://www.askpython.com/python/list/python-list)，它是可变的。

```py
a = [1, 2, 3, 4]

b = a

print(a)
print(b)

b.append(5)

# Changes will be reflected in a too!
print(a)
print(b)

```

**输出**

```py
[1, 2, 3, 4]
[1, 2, 3, 4]
[1, 2, 3, 4, 5]
[1, 2, 3, 4, 5]

```

如你所见，由于两个变量指向同一个对象，所以当`b`改变时，`a`也会改变！

为了处理这个问题，Python 给了我们一种使用 *Copy* 模块的方法。

Python 复制模块是标准库的一部分，可以使用以下语句导入:

```py
import copy

```

现在，在本模块中，我们主要可以执行两种类型的操作:

*   浅拷贝
*   深层拷贝

现在让我们来看看这些方法。

* * *

## 浅拷贝

此方法用于执行浅层复制操作。

调用此方法的语法是:

```py
import copy

new_obj = copy.copy(old_obj) # Perform a shallow copy

```

这将做两件事——

*   创建新对象
*   插入在原始对象中找到的对象的所有引用

现在，由于它创建了一个新对象，我们可以确定我们的新对象不同于旧对象。

但是，这仍将保持对嵌套对象的引用。因此，如果我们需要复制的对象有其他可变对象(列表、集合等)，这仍然会维护对同一个嵌套对象的引用！

为了理解这一点，我们举个例子。

为了说明第一点，我们将尝试使用一个简单的整数列表(没有嵌套对象！)

```py
import copy

old_list = [1, 2, 3]

print(old_list)

new_list = copy.copy(old_list)

# Let's try changing new_list
new_list.append(4)

# Changes will not be reflected in the original list, since the objects are different
print(old_list)
print(new_list)

```

**输出**

```py
[1, 2, 3]
[1, 2, 3, 4]
[1, 2, 3]

```

如你所见，如果我们的对象是一个简单的列表，那么浅拷贝就没有问题。

让我们看另一个例子，我们的对象是一个列表列表。

```py
import copy

old_list = [[1, 2], [1, 2, 3]]

print(old_list)

new_list = copy.copy(old_list)

# Let's try changing a nested object inside the list
new_list[1].append(4)

# Changes will be reflected in the original list, since the object contains a nested object
print(old_list)
print(new_list)

```

**输出**

```py
[[1, 2], [1, 2, 3]]
[[1, 2], [1, 2, 3, 4]]
[[1, 2], [1, 2, 3, 4]]

```

这里，请注意`old_list`和`new_list`都受到了影响！

如果我们必须避免这种行为，我们必须递归地复制所有对象，以及嵌套的对象。

这被称为使用 Python 复制模块的*深度复制*操作。

* * *

## 深层拷贝

此方法类似于浅层复制方法，但现在将原始对象中的所有内容(包括嵌套对象)复制到新对象中。

要执行深度复制操作，我们可以使用以下语法:

```py
import copy

new_object = copy.deepcopy(old_object)

```

让我们以以前的例子为例，尝试使用深层拷贝来解决我们的问题。

```py
import copy

old_list = [[1, 2], [1, 2, 3]]

print(old_list)

new_list = copy.deepcopy(old_list)

# Let's try changing a nested object inside the list
new_list[1].append(4)

# Changes will be reflected in the original list, since the objects are different
print(old_list)
print(new_list)

```

**输出**

```py
[[1, 2], [1, 2, 3]]
[[1, 2], [1, 2, 3]]
[[1, 2], [1, 2, 3, 4]]

```

请注意，旧列表没有改变。因为所有对象都是递归复制的，所以现在没有问题了！

但是，由于要复制所有对象，与浅层复制方法相比，这种深层复制方法的成本要高一些。

所以明智地使用它，只在你需要的时候！

* * *

## 结论

在本文中，我们学习了如何使用 Python 复制模块来执行浅层复制和深层复制操作。

## 参考

*   Python 复制模块[文档](https://docs.python.org/3.8/library/copy.html)
*   关于 Python 复制模块的 JournalDev 文章

* * *