# Python 中 Pop 和 Remove 的区别

> 原文：<https://www.pythonforbeginners.com/basics/difference-between-pop-and-remove-in-python>

在各种任务中，我们需要从列表中删除或提取元素。我们通常使用`pop()`方法和`remove()`方法来实现。在本文中，我们将讨论 python 中 pop()方法和 remove()方法的主要区别。

## pop()方法

方法用来从一个给定的列表中提取一个元素。当在列表上调用时，它接受元素的`index`作为可选的输入参数，并在从列表中删除元素后返回给定索引处的元素，如下所示。

```py
myList = [1, 2, 3, 4, 5, 6, 7]
print("The original list is:", myList)
element = myList.pop(2)
print("The popped element is:", element)
print("The updated list is:", myList)
```

输出:

```py
The original list is: [1, 2, 3, 4, 5, 6, 7]
The popped element is: 3
The updated list is: [1, 2, 4, 5, 6, 7]
```

如果我们不提供任何索引作为输入参数，它将删除最后一个索引处的元素并返回值。

```py
myList = [1, 2, 3, 4, 5, 6, 7]
print("The original list is:", myList)
element = myList.pop()
print("The popped element is:", element)
print("The updated list is:", myList)
```

输出:

```py
The original list is: [1, 2, 3, 4, 5, 6, 7]
The popped element is: 7
The updated list is: [1, 2, 3, 4, 5, 6]
```

## remove()方法

方法也被用来从列表中删除一个元素。在列表上调用`remove()`方法时，该方法将需要删除的元素的值作为输入参数。执行后，它从列表中删除输入元素的第一个匹配项。您可以在下面的示例中观察到这一点。

```py
myList = [1, 2, 3, 4, 5, 6, 7]
print("The original list is:", myList)
myList.remove(3)
print("The removed element is:", 3)
print("The updated list is:", myList)
```

输出:

```py
The original list is: [1, 2, 3, 4, 5, 6, 7]
The removed element is: 3
The updated list is: [1, 2, 4, 5, 6, 7]
```

`remove()`方法不返回任何值。

## Pop 和 Remove 的区别

*   `pop()`方法和`remove()`方法的主要区别在于,`pop()` 方法使用元素的索引来删除它，而`remove()`方法将元素的值作为输入参数来删除元素，正如我们在上面已经看到的。
*   可以在没有输入参数的情况下使用`pop()`方法。另一方面，如果我们使用没有输入参数的`remove()`方法，程序将会出错。
*   `pop()`方法返回被删除元素的值。然而，`remove()`方法不返回任何值。
*   如果我们在一个空列表上调用`pop()`方法，它将引发一个`IndexError`异常。另一方面，如果我们在一个空列表上调用`remove()`方法，它将引发`ValueError`异常。

## 结论

在本文中，我们讨论了 python 中列表的 pop()方法和 remove()方法之间的区别。要了解更多关于列表的知识，你可以阅读这篇关于 python 中的[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。你可能也会喜欢这篇关于 python 中的[集合理解的文章。](https://www.pythonforbeginners.com/basics/set-comprehension-in-python)

建议阅读:

*   [用 Python 开发聊天应用，源代码](https://codinginfinite.com/python-chat-application-tutorial-source-code/)
*   [使用 Python 中的 sklearn 模块进行多项式回归](https://codinginfinite.com/polynomial-regression-using-sklearn-module-in-python/)