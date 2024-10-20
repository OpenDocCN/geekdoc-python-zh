# 如何使用 Python List pop()方法

> 原文：<https://www.askpython.com/python/list/python-list-pop>

Python list `pop()`方法用于从 Python 列表中弹出项目。在本文中，我们将快速看一下如何使用`pop()`从[列表](https://www.askpython.com/python/list/python-list)中弹出元素。

* * *

## Python 列表 pop()的基本语法

这是一个列表对象类型的方法，所以每个列表对象都有这个方法。

**您可以通过使用:**来调用它

```py
my_list.pop()

```

这是默认的调用，只是从列表中弹出最后一项。

如果您想从索引中弹出一个元素，我们也可以传递索引。

```py
last_element = my_list.pop(index)

```

这将在`index`弹出元素，并相应地更新我们的列表。它将返回弹出的元素，但在大多数情况下，您可以选择忽略返回值。

现在我们已经讨论了语法，让我们看看如何使用它。

* * *

## 使用 Python list.pop()

让我们来看一下默认情况，您只需要弹出最后一个元素。

```py
# Create a list from 0 to 10
my_list = [i for i in range(11)]

# Pop the last element
print("Popping the last element...")
my_list.pop()

# Print the modified list
print("List after popping:", my_list)

# Again Pop the last element
print("Popping the last element...")
my_list.pop()

# Print the modified list
print("List after popping:", my_list)

```

**输出**

```py
Popping the last element...
List after popping: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Popping the last element...
List after popping: [0, 1, 2, 3, 4, 5, 6, 7, 8]

```

如您所见，最后一个元素确实从我们的列表中弹出。

现在，让我们考虑第二种类型，您希望在特定索引处弹出元素。

```py
# Create a list from 0 to 10
my_list = [i for i in range(11)]

# Pop the last element
print("Popping the element at index 5...")
my_list.pop(5)

# Print the modified list
print("List after popping:", my_list)

# Again Pop the last element
print("Popping the element at index 2...")
my_list.pop(2)

# Print the modified list
print("List after popping:", my_list)

```

**输出**

```py
Popping the element at index 5...
List after popping: [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
Popping the element at index 2...
List after popping: [0, 1, 3, 4, 6, 7, 8, 9, 10]

```

在这里，由于我们的原始列表是`[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`，索引 5 处的元素是`list[5]`，也就是`5`。所以，这个被删除了，我们的列表现在没有了 5。类似地，在新列表中，我们再次删除第二个索引处的元素，即`2`。因此，我们的最终名单是`[0, 1, 3, 4, 6, 7, 8, 9, 10]`。

## 处理异常

如果违反了某些条件，`list.pop()`方法将引发一些[异常](https://www.askpython.com/python/python-exception-handling)。

### 列表为空时出现 IndexError 异常

当使用 Python list pop()方法时，如果我们的列表为空，我们就不能再从中弹出。这将引发一个`IndexError`异常。

```py
my_list = []

# Will raise an exception, since the list is empty
my_list.pop()

```

**输出**

```py
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: pop from empty list

```

因为我们试图从一个空列表中弹出，所以引发了这个异常，并显示了相应的错误消息。

### 索引时出现索引错误异常

如果传递给`pop(index)`方法的索引超出了列表的大小，就会引发这个异常。

例如，试图删除包含 11 个元素的列表中的第 12 个索引元素将引发此异常。

```py
my_list = [i for i in range(10)]

# Will raise an exception, as len(my_list) = 11
my_list.pop(12)

```

输出

```py
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
IndexError: pop index out of range

```

正如所料，我们确实得到了 IndexError 异常，因为`my_list[12`不存在。

* * *

## 结论

在本文中，我们学习了如何使用`list.pop()`方法从列表中弹出元素。

## 参考

*   关于 Python List pop()方法的 JournalDev 文章

* * *