# 在 Python 中追加到列表

> 原文：<https://www.askpython.com/python/list/append-to-a-list-in-python>

在本文中，我们将看看如何在 Python 中向一个[列表](https://www.askpython.com/python/list/python-list)追加内容。

Python 的 **list.append()** 提供了这个问题的解决方案，所以我们将看到一些使用这个方法的例子。

我们开始吧！

* * *

## 在 Python 中追加到普通列表

我们可以在列表中使用 Python 内置的 **append()** 方法，并将元素添加到列表的末尾。

```py
my_list = [2, 4, 6, 8]

print("List before appending:", my_list

# We can append an integer
my_list.append(10)

# Or even other types, such as a string!
my_list.append("Hello!")

print("List after appending:", my_list)

```

**输出**

```py
List before appending: [2, 4, 6, 8]
List after appending: [2, 4, 6, 8, 10, "Hello!"]

```

如您所见，我们的列表在末尾插入了两个元素 **10** 和“**你好**”。当你追加到一个普通列表时就是这种情况。

现在让我们看看其他一些案例。

* * *

## 追加到 Python 嵌套列表中的列表

嵌套列表是一个包含其他列表的列表。在这个场景中，我们将了解当列表嵌套时，如何在 Python 中追加列表。

我们将看看嵌套列表有不同长度的 **N** 个列表的特殊情况。我们想在原来的列表中插入另一个正好包含 **N** 个元素的列表。

但是现在，我们不是直接追加到嵌套列表，而是将每个 **N** 元素依次追加到每个 **N** 列表。

为了给你看一个例子，这里是我们的嵌套列表，有 **N = 3** 个列表:

```py
nested_list = [[1, 2, 3], [4, 5, 6, 7], [2, 4, 5, 6, 7]]

```

我们将插入列表的 N 个元素中的每一个:

```py
my_list = [10, 11, 12]

```

10 将被添加到第一个列表中，11 将被添加到第二个列表中，12 将被添加到第三个列表中。

因此，我们的输出将是:

```py
[[1, 2, 3, 10], [4, 5, 6, 7, 11], [2, 4, 5, 6, 7, 12]]

```

有问题吗？现在就来解决吧！

因此，对于嵌套列表中的每个列表，我们从`my_list`中选择相应的元素，并将其附加到该列表中。我们一直这样做，直到到达嵌套列表的末尾，以及`my_list`。

一种可能的方法是遍历嵌套列表。因为我们知道嵌套列表的每个元素都是一个列表，所以我们可以获取当前元素的索引，并将`my_list[idx]`附加到`nested_list[idx]`。

```py
nested_list = [[1, 2, 3], [4, 5, 6, 7], [2, 4, 5, 6, 7]]

my_list = [10, 11, 12]

for idx, small_list in enumerate(nested_list):
    small_list.append(my_list[idx])

print(nested_list)

```

**输出**

```py
[[1, 2, 3, 10], [4, 5, 6, 7, 11], [2, 4, 5, 6, 7, 12]]

```

的确，我们的产量符合我们的预期！

* * *

## 结论

在本文中，我们学习了如何追加到 Python 列表中，并研究了这个过程的各种情况。

* * *