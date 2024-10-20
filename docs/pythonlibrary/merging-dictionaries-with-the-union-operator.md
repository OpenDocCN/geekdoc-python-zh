# 用 Union 运算符合并字典

> 原文：<https://www.blog.pythonlibrary.org/2021/09/18/merging-dictionaries-with-the-union-operator/>

作为一名开发人员，有时您可能需要将两个或更多的字典合并成一个主字典。在 Python 编程语言中，有许多不同的方式来合并字典。

在本教程中，你将看到一些合并字典的老方法，然后再看看在 **Python 3.9** 中添加的最新方法。

以下是您将了解的方法:

*   使用 dict.update()
*   与**合并
*   用 Union 运算符合并

您将使用 **update()** 方法开始您的旅程！

## 使用 dict.update()

Python 的字典有很多不同的方法。这些方法中的一种可以用来将两个字典合并在一起。这个方法叫做 **update()** 。

这里有一个例子:

```py
>>> first_dictionary = {"name": "Mike", "occupation": "Python Teacher"}
>>> second_dictionary = {"location": "Iowa", "hobby": "photography"}
>>> first_dictionary.update(second_dictionary)
>>> first_dictionary
{'name': 'Mike', 'occupation': 'Python Teacher', 'location': 'Iowa', 'hobby': 'photography'}

```

这工作完美！这个方法的唯一问题是它修改了其中一个字典。如果您想要创建第三个字典，而不修改其中一个输入字典，那么您将需要查看本文中的其他合并方法。

您现在已经准备好学习使用**！

## 与**合并

当你使用双星号时，它有时被称为“打开”、“展开”或“打开”字典。Python 中使用了 ****** ，函数中也使用了 **kwargs** 。

下面是如何使用**来合并两个字典:

```py
>>> first_dictionary = {"name": "Mike", "occupation": "Python Teacher"}
>>> second_dictionary = {"location": "Iowa", "hobby": "photography"}
>>> merged_dictionary = {**first_dictionary, **second_dictionary}
>>> merged_dictionary
{'name': 'Mike', 'occupation': 'Python Teacher', 'location': 'Iowa', 'hobby': 'photography'}
```

这种语法看起来有点奇怪，但它非常有效！

现在，您已经准备好了解合并两个词典的最新方法了！

## 用 Union 运算符合并

从 **Python 3.9** 开始，可以使用 Python 的 union 操作符 **|** 来合并字典。你可以在 [PEP 584](https://www.python.org/dev/peps/pep-0584/) 中了解所有的实质细节。

下面是如何使用 union 运算符合并两个字典:

```py
>>> first_dictionary = {"name": "Mike", "occupation": "Python Teacher"} 
>>> second_dictionary = {"location": "Iowa", "hobby": "photography"}
>>> merged_dictionary = first_dictionary | second_dictionary
>>> merged_dictionary
{'name': 'Mike', 'occupation': 'Python Teacher', 'location': 'Iowa', 'hobby': 'photography'}
```

这是将两本词典合二为一的最短方法。

## 包扎

您现在知道了三种不同的方法，可以用来将多个字典合并成一个。如果你有 Python 3.9 或更高版本的权限，你应该使用 union 操作符，因为这可能是看起来最干净的合并字典的方法。然而，如果你被困在一个旧版本的 Python 上，你不必绝望，因为你现在有另外两个方法可以工作！