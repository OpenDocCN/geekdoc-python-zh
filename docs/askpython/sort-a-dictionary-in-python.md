# 如何用 Python 对字典进行排序？

> 原文：<https://www.askpython.com/python/dictionary/sort-a-dictionary-in-python>

在 Python 中，我们可以通过不同的方式对字典进行排序。有很多种方法，取决于你是想按键排序还是按值排序。

让我们来看看文章中的一些吧！

* * *

## 按值对 Python 中的字典进行排序

如果我们想按值对 Python 中的字典进行排序，有几种方法可以实现。

### 方法 1:使用 lambda 使用 sorted()(Python 3.6+推荐)

在 Python 的新版本上( **Python 3.6** 和更高版本)，我们可以在`sorted()`方法上使用 lambda 来对 Python 中的字典进行排序。

我们将使用关键字作为 lambda 对`dict.items()`进行排序。

```py
my_dict = {1: 2, 2: 10, "Hello": 1234}

print({key: value for key, value in sorted(my_dict.items(), key=lambda item: item[1])})

```

**输出**

```py
{1: 2, 2: 10, 'Hello': 1234}

```

的确，我们可以看到我们的新字典已经根据值进行了排序！

这种方法在 Python 3.6+上有效的原因是 Python 新版本中的字典现在是有序的数据类型。

这意味着我们可以将字典枚举为条目列表，还可以执行改变顺序的操作，比如排序。

但是不要害怕。如果你有旧版本的 Python，请继续阅读。我们将向您展示另一种处理方法！

### 方法 2:在旧版本的 Python 上使用 sorted()

我们仍然可以使用`sorted()`对字典进行排序。但是我们需要把字典整理成一个有序的类型。`operator`模块有，使用`operator.itemgetter(idx)`。

下面的代码片段将根据值对我们的字典进行排序:

```py
import operator

my_dict = {1: 2, 2: 10, "Hello": 1234}

sorted_dict = sorted(my_dict.items(), key=operator.itemgetter(1))

print(sorted_dict)

```

更具体地说，我们使用`sorted(dict.items()`形成一个排序列表，并将`operator.itemgetter(1)`传递给它(因为值在索引 1 处)。

这将构造一个 callable，从 items 列表中获取第一个元素。我们在每次迭代中都这样做，从而得到一个排序的字典！

* * *

## 按关键字对 Python 中的字典进行排序

### 方法 1:使用 operator.itemgetter()(旧版本 Python 的推荐方法)

现在，如果你想按键对字典排序，我们可以使用`operator`方法，就像上一节一样。我们必须做的唯一改变是现在根据键对列表进行排序，所以我们调用`operator.itemgetter(0)`。

```py
import operator

my_dict = {2: 10, 1: 2, -3: 1234}

# Sort the Dict based on keys
sorted_dict = dict(sorted(my_dict.items(), key=operator.itemgetter(0)))

print(sorted_dict)

```

**输出**

```py
{-3: 1234, 1: 2, 2: 10}

```

的确，字典现在已经根据键排序了！

### 方法 2:使用 sorted()和 lambda(Python 3.6+的推荐方法)

我们可以在新版本的 Python 上使用带有 lambda 的`sorted()`方法。

同样，这与之前相同，但我们现在将基于值进行排序。

```py
my_dict = {2: 10, 1: 2, -3: 1234}
sorted_dict = dict(sorted(my_dict.items(), key=lambda item: item[0]))

```

**输出**

```py
{-3: 1234, 1: 2, 2: 10}

```

同样，输出与之前相同。

* * *

## 结论

在本文中，我们学习了如何用 Python 对字典进行排序；通过键和值，使用不同的方法。

* * *

## 参考

*   关于按值排序字典的问题

* * *