# 在 Python 中将字符串转换为集合

> 原文：<https://www.pythonforbeginners.com/basics/convert-string-to-set-in-python>

在 python 中，字符串用于操作文本数据。有时，我们可能需要找出文本中不同字符的总数。在这种情况下，我们可以将字符串转换成集合。在本文中，我们将讨论在 python 中将字符串转换为集合的不同方法。

## 使用 Set()函数在 Python 中将字符串转换为集合

`set()`函数用于在 Python 中创建一个集合。它将一个 iterable 对象作为其输入参数，并返回一个包含 iterable 对象中元素的集合。

正如我们所知，字符串是一个可迭代的对象，我们可以使用`set()`函数从 python 中的字符串获得一个集合。为此，我们将把字符串作为输入参数传递给`set()`函数。执行`set()`函数后，我们将得到输入字符串的所有字符的集合。您可以在下面的示例中观察到这一点。

```py
myStr = "pythonforbeginners"
mySet = set(myStr)
print("The input string is:", myStr)
print("The output set is:", mySet)
```

输出:

```py
The input string is: pythonforbeginners
The output set is: {'f', 'i', 'g', 'y', 'o', 'r', 't', 'h', 'p', 'n', 'b', 'e', 's'}
```

在上面的例子中，您可以观察到我们将字符串`pythonforbeginners`作为输入传递给了`set()`函数。执行后，它返回一个包含字符串中字符的集合。

## 使用集合理解在 Python 中将字符串转换为集合

[集合理解](https://www.pythonforbeginners.com/basics/set-comprehension-in-python)用于从现有的可迭代对象创建新的集合。集合理解的语法如下。

```py
newSet= { expression for element in  iterable }
```

为了使用 Python 中的集合理解将字符串转换成集合，我们将使用输入字符串作为`iterable`，将字符串的字符作为`element`以及`expression`。执行后，上述语句会将字符串转换为 set。您可以在下面的示例中观察到这一点。

```py
myStr = "pythonforbeginners"
mySet = {character for character in myStr}
print("The input string is:", myStr)
print("The output set is:", mySet)
```

输出:

```py
The input string is: pythonforbeginners
The output set is: {'i', 'n', 'o', 's', 't', 'h', 'b', 'g', 'e', 'r', 'p', 'y', 'f'}
```

在上面的例子中，我们已经使用 set comprehension 将`pythonforbeginners`转换为 Python 中的集合。

## 使用 add()方法在 Python 中设置的字符串

`add()`方法用于向集合中添加一个元素。当在集合上调用时，`add()`方法将一个元素作为它的输入参数。执行后，如果元素不存在于集合中，它会将该元素添加到集合中。如果元素已经存在于集合中，则什么也不会发生。

要使用 python 中的`add()`方法将字符串转换为集合，我们将使用以下步骤。

*   首先，我们将创建一个名为`mySet`的空集。为此，我们将使用`set()`函数。`set()`函数在不带任何参数的情况下执行时，返回一个空集。
*   创建空集后，我们将使用 for 循环遍历输入字符串的字符。
*   在迭代过程中，我们将调用`mySet`上的`add()`方法，并将每个字符添加到`mySet`。
*   在执行 for 循环后，我们将获得变量`mySet`中设置的输出。您可以在下面的示例中观察到这一点。

```py
myStr = "pythonforbeginners"
mySet = set()
for character in myStr:
    mySet.add(character)
print("The input string is:", myStr)
print("The output set is:", mySet)
```

输出:

```py
The input string is: pythonforbeginners
The output set is: {'h', 'n', 'g', 'f', 'i', 'b', 'o', 'p', 't', 'e', 's', 'r', 'y'}
```

在这个例子中，我们使用了`add()`方法将字符串 pythonforbeginners 转换为一个集合。

## 结论

在本文中，我们讨论了在 Python 中将字符串转换为集合的三种方法。在所有这三种方法中，如果您需要在集合中包含字符串的所有字符，您可以使用使用`set(`函数的方法。

如果您需要从集合中排除输入字符串的某些字符，您可以使用带有`add()`方法或集合理解的方法。要了解更多关于 python 编程的知识，你可以阅读这篇关于 python 中的[字符串连接](https://www.pythonforbeginners.com/concatenation/string-concatenation-and-formatting-in-python)的文章。你可能也会喜欢这篇关于 Python 中[字典理解的文章。](https://www.pythonforbeginners.com/dictionary/dictionary-comprehension-in-python)

我希望你喜欢阅读这篇文章。请继续关注更多内容丰富的文章。

快乐学习！