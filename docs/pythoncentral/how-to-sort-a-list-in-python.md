# 如何在 Python 中对列表进行排序

> 原文：<https://www.pythoncentral.io/how-to-sort-a-list-in-python/>

每个程序员都必须编写在某些时候涉及到数据排序的代码。随着 Python 技能的进步，您可能会发现自己正在处理一个排序问题，这个问题对于应用程序中的用户体验至关重要。

您可能需要编写代码来通过时间戳跟踪用户活动。您可能还需要编写一个脚本，按照字母顺序排列电子邮件收件人。

Python 拥有内置的排序功能，这使得程序员可以很容易地对不同类型的数据进行排序。该语言还包括一些特性，使程序员能够在粒度级别定制排序操作。

这里有一个如何在 Python 中对列表排序的快速指南。

## **如何在 Python 中对列表进行排序**

Python 有两个内置函数可以帮助你对列表进行排序:sort()和 sorted()。这两个函数使用相同的参数。但是，这两个函数具有不同的语法，并且与 iterables 的交互方式也不同。

也就是说，学会使用这两个功能会让列表排序变得轻而易举。如果你不熟悉用 Python 创建一个列表，在阅读之前先浏览一下我们的 [简易指南](https://www.pythoncentral.io/lists-in-python-how-to-create-a-list-in-python/) 是个不错的主意。

下面是如何使用这两种排序方法的详细分析。

## **分类方法**

Python list sort()方法允许程序员按照升序和降序对数据进行排序。默认情况下，该方法将按升序对列表进行排序。

程序员也可以编写一个函数来定制排序标准。

排序方法的语法是:

```py
*list*.sort(reverse=True|False, key=myFunc)
```

其中“列表”必须替换为存储数据的变量名。sort()函数不返回任何值。它直接对列表中的值进行排序，而不创建新的变量来存储排序后的列表。

下面是一个如何使用 sort()函数对数据进行升序排序的例子:

```py
names = ['Liam', 'Noah', 'Emma']
names.sort()

print(names)

```

输出将是:

```py
['Emma','Liam','Noah']
```

### 因素

sort()函数不需要任何参数就能工作。但是，如果您想改变数据的排序方式，可以使用以下参数:

*   **键:** 该参数作为排序操作的键。
*   **反转:** 如果参数设置为“真”，列表将按降序排序。

下面是一个使用 reverse 参数对列表进行降序排序的例子:

```py
names = ['Liam', 'Noah', 'Emma']
names.sort(reverse=True)

print(names)
```

代码的输出是:

```py
['Noah', 'Liam', 'Emma']
```

您可以使用 key 参数根据长度对列表进行排序。下面是一个执行此操作的脚本示例:

```py
def myFunc(e):
  return len(e)

names = ['Emily', 'Fred', 'Robert']
names.sort(key=myFunc)
print(names)
```

脚本的输出是:

```py
['Fred', 'Emily', 'Robert']
```

## **排序方法**

另一种在 Python 中对列表进行升序或降序排序的方法是使用 sorted()方法。像 sort()一样，它不需要定义，因为它是一个内置函数。Python 的每个标准版本都带有内置的方法。

此外，它可以不带任何参数使用，类似于 sort()。默认情况下，该函数按升序排列数据。

然而，与 sort()不同的是，sorted()函数将排列后的值存储在一个新的列表中。应用该函数后，原始列表保持不变。相反，sort()函数返回“无”

因此，由函数返回的数据的有序列表可以在函数执行后被分配给变量。

Python 中可以迭代的对象称为 iterables。sorted()方法适用于所有的[](https://www.pythonlikeyoumeanit.com/Module2_EssentialsOfPython/Iterables.html)。换句话说，除了列表，sorted()还可以处理字符串、元组和集合。

该方法的语法为:

```py
sorted(iterable, key, reverse)
```

使用 sorted()的另一个优点是，您可以将值传递给它，而不必先创建一个变量并在其中存储值。

例如:

```py
sorted([10, 9, 8, 7, 6])
#result = [6,7,8,9,10]

```

这并不是说 sorted()函数不能处理变量:

```py
>>> exampleList = [7, 9, 2, 4]
>>> sorted(exampleList)
[2, 4, 7, 9]
>>> exampleList
[7, 9, 2, 4]
# Values of exampleList did not change. New list was returned.

```

### **参数**

sorted()方法与 sort()共享参数。使用“reverse”参数将按降序输出一个列表，您可以使用“key”参数定义排序顺序。

下面是一个使用 sorted()方法进行降序排序的例子:

```py
>>> names = ['Liam', 'Noah', 'Emma']
>>> sorted(names)
['Emma', 'Liam', 'Noah']
>>> sorted(names, reverse=True)
['Noah', 'Liam', 'Emma']
```

排序逻辑没有改变——名字仍然按照第一个字母排序。但是，由于参数设置为“真”，输出已经反转如果参数设置为“False”，结果将保持不变。

“key”参数是 sorted()方法中最强大的组件。它期望传递一个函数，然后对列表中的每个值使用该函数来导出数据的顺序。

例如，您可以使用关键字参数根据单词的长度对单词列表进行排序

```py
>>> exampleWords = ['cat', 'mouse', 'chicken', 'bird']
>>> sorted(exampleWords, key=len)
['cat', 'bird', 'mouse', 'chicken']
```

要按长度降序排列上面的列表，可以将 key 参数与 reverse 参数结合使用。

```py
>>> exampleWords = ['cat', 'mouse', 'chicken', 'bird']
>>> sorted(exampleWords, key=len, reverse=True)
['chicken', 'mouse', 'bird', 'cat']
```

有时，您可以使用 lambda 函数，这是一个内置函数，而不是编写一个独立的函数来使用 key 参数。

匿名功能有四个特点:

1.  必须内联定义；
2.  它不能包含语句；
3.  它没有名字；和
4.  它就像一个函数一样执行。

假设一个程序员想写一个程序，根据数字的绝对值对它们进行排序。

这个脚本看起来应该是这样的:

```py
num_list = [1,-5,3,-9,25,10]
def absolute_value(num):
   return abs(num)
num_list.sort(key = absolute_value)
print(num_list)
```

但是，我们可以使用下面的代码行，而不是编写 absolute_value 函数并将其分配给 key 参数:

```py
num_list.sort(key = lambda num: abs(num))
```

# **结论**

选择使用哪种方法对列表进行排序可能会令人困惑。这里有一个选择方法的好方法:

如果你不需要原始列表，使用 sort()是正确的方法。它不创建新的列表，因此使用更少的内存并提高代码的效率。

然而，如果你认为你以后需要访问原始列表，你必须使用 sorted()函数。

Python 能帮你做的不仅仅是排序数据。阅读 [我们的使用指南](https://www.pythoncentral.io/what-can-you-do-with-python-usage-guide/) 是了解 Python 所有功能的好方法。