# Python 过滤器()函数

> 原文：<https://www.askpython.com/python/built-in-methods/python-filter-function>

Python 的`**filter**()`函数用于在测试 iterable 上每个元素的谓词的帮助下过滤 iterable(序列)的元素。

谓词是一个总是返回`**True**`或`**False**`的函数。我们不能将通用函数与`filter()`一起使用，因为只有满足合适的条件时，它才返回所有元素。这意味着过滤函数必须总是返回一个布尔值，因此，过滤函数是一个谓词。

* * *

## 过滤器的基本格式()

因为这是一个在 Python iterable 上操作的函数，所以 iterable 是参数之一。因为它在每个元素上测试一个谓词，所以这个函数也是另一个需要的参数。

因为它从序列中过滤出元素，所以它还必须返回一个 iterable，该 iterable 只包含满足过滤函数的元素。

但是在这种情况下，由于我们使用的是对象，Python 返回给我们一个 ***过滤器对象*** 作为可迭代对象，这将证明使用类似`list()`和`dict()`的方法转换成其他类型是很方便的。

很简单，不是吗？让我们看看我们如何应用它，并使用`filter()`创建工作程序。

格式:`filter_object = filter(predicate, iterable)`

下面是一个非常简单的例子，用一个函数过滤一个列表，这个函数测试一个数字是奇数还是偶数。

```py
a = [1, 2, 3, 4, 5]

# We filter using a lambda function predicate.
# This predicate returns true
# only if the number is even.
filter_obj_even = filter(lambda x: x%2 == 0, a)

print(type(filter_obj_even))

# Convert to a list using list()
print('Even numbers:', list(filter_obj_even))

# We can also use define the predicate using def()
def odd(num):
    return (num % 2) != 0

filter_obj_odd = filter(odd, a)
print('Odd numbers:', list(filter_obj_odd))

```

输出

```py
<class 'filter'>
Even numbers: [2, 4]
Odd numbers: [1, 3, 5]

```

请注意，我们可以通过迭代获得 filter 对象的单个元素，因为它是可迭代的:

```py
for item in filter_obj_odd:
    print(item)

```

输出

```py
1
3
5

```

* * *

## 过滤器()和无

我们也可以用`None`和`filter()`做谓语。如果对象的布尔值为`True`，则`None`返回`True`，否则返回`False`。

这意味着像`0`、`None`、`''`、`[]`等对象都被`None`谓词过滤掉，因为它们是空元素对象。

```py
a = [0, 1, 'Hello', '', [], [1,2,3], 0.1, 0.0]

print(list(filter(None, a)))

```

输出

```py
[1, 'Hello', [1, 2, 3], 0.1]

```

* * *

## 结论

我们学习了 Python 为我们提供的用于在 iterable 上应用谓词的`filter()`函数。

`filter`的简洁性和可读性使其成为现代 Python 代码库开发人员中非常受欢迎的函数。

* * *

## 参考

*   关于 Python 过滤器的 JournalDev 文章
*   [Python.org API 文件](https://docs.python.org/3.8/library/functions.html#filter)