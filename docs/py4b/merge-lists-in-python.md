# Python 中的合并列表

> 原文：<https://www.pythonforbeginners.com/lists/merge-lists-in-python>

在编程时，我们可能需要在 python 中合并两个或多个列表。在本文中，我们将看看在 python 中合并两个列表的不同方法。

## 使用 python 中的 append()方法合并列表

我们可以使用 append()方法将一个列表合并到另一个列表中。append()方法用于向现有列表添加新元素。要使用 append()方法合并两个列表，我们将获取一个列表，并使用 for 循环将另一个列表中的元素逐个添加到该列表中。这可以如下进行。

```py
list1=[1,2,3,4]
list2=[5,6,7,8]
print("First list is:")
print(list1)
print("Second list is:")
print(list2)
for i in list2:
    list1.append(i)
print("Merged list is:")
print(list1)
```

输出:

```py
First list is:
[1, 2, 3, 4]
Second list is:
[5, 6, 7, 8]
Merged list is:
[1, 2, 3, 4, 5, 6, 7, 8]
```

在使用 append()方法合并列表的过程中，随着第二个输入列表中的元素被添加到第一个输入列表中，第一个输入列表被修改。而对第二个列表没有影响，我们从第二个列表中取出元素添加到第一个列表中。

## 使用 append()和 pop()方法

除了 append()方法，我们还可以使用 pop()方法在 python 中合并两个列表。对任何列表调用 pop()方法时，都会删除最后一个元素并返回它。我们将使用 pop()方法从一个列表中取出元素，并使用 append()方法向另一个列表中添加元素。这可以如下进行。

```py
list1=[1,2,3,4]
list2=[5,6,7,8]
print("First list is:")
print(list1)
print("Second list is:")
print(list2)
while list2:
    temp=list2.pop()
    list1.append(temp)
print("Merged list is:")
print(list1)
```

输出:

```py
First list is:
[1, 2, 3, 4]
Second list is:
[5, 6, 7, 8]
Merged list is:
[1, 2, 3, 4, 8, 7, 6, 5]
```

当使用上述方法合并两个列表时，两个输入列表都会被修改。当其他列表中的元素被添加到第一个列表中时，第一个输入列表被修改。使用 pop()方法删除第二个列表中的所有元素，因此在合并列表后，第二个列表变为空。在输出中，我们还可以看到第二个列表中的元素在合并列表中以相反的顺序出现。

## 使用列表理解

我们还可以使用[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)来合并 python 中的两个或更多列表。为此，我们将首先创建一个要合并的所有列表的列表，然后我们可以使用列表理解来创建合并的列表，如下所示。

```py
list1=[1,2,3,4]
list2=[5,6,7,8]
print("First list is:")
print(list1)
print("Second list is:")
print(list2)
combined_list=[list1,list2]
merged_list=[item for sublist in combined_list for item in sublist]
print("Merged list is:")
print(merged_list)
```

输出:

```py
First list is:
[1, 2, 3, 4]
Second list is:
[5, 6, 7, 8]
Merged list is:
[1, 2, 3, 4, 5, 6, 7, 8]
```

当我们使用列表理解合并列表时，没有一个要被合并的输入列表会被修改。

## 在 python 中使用+运算符合并列表

我们可以使用+运算符直接合并两个或多个列表，只需使用+运算符将所有列表相加即可，如下所示。

```py
list1=[1,2,3,4]
list2=[5,6,7,8]
print("First list is:")
print(list1)
print("Second list is:")
print(list2)
merged_list=list1+list2
print("Merged list is:")
print(merged_list)
```

输出:

```py
First list is:
[1, 2, 3, 4]
Second list is:
[5, 6, 7, 8]
Merged list is:
[1, 2, 3, 4, 5, 6, 7, 8]
```

当我们使用上面程序中的+操作符合并两个列表时，被合并的输入列表都不会被修改。

## 使用星号运算符

我们可以使用打包操作符*将两个或多个列表打包在一起，以合并列表，如下所示。

```py
list1=[1,2,3,4]
list2=[5,6,7,8]
print("First list is:")
print(list1)
print("Second list is:")
print(list2)
merged_list=[*list1,*list2]
print("Merged list is:")
print(merged_list)
```

输出:

```py
First list is:
[1, 2, 3, 4]
Second list is:
[5, 6, 7, 8]
Merged list is:
[1, 2, 3, 4, 5, 6, 7, 8]
```

当我们使用星号操作符合并两个列表时，被合并的输入列表都不会被修改。这种方法对于一次合并三个或更多列表也很方便，因为我们可以通过使用*操作符简单地提到每个列表，在输出的合并列表中包含每个列表的元素。

## 使用 extend()方法

要使用 extend()方法合并两个或多个列表，我们可以通过添加其他列表的元素来执行第一个列表的就地扩展，如下所示。

```py
list1=[1,2,3,4]
list2=[5,6,7,8]
print("First list is:")
print(list1)
print("Second list is:")
print(list2)
list1.extend(list2)
print("Merged list is:")
print(list1)
```

输出:

```py
First list is:
[1, 2, 3, 4]
Second list is:
[5, 6, 7, 8]
Merged list is:
[1, 2, 3, 4, 5, 6, 7, 8]
```

当我们使用 extend()方法合并两个列表时，extend()方法所在的列表会随着其他列表中的元素被添加到其中而被修改。其他列表不受该操作的影响。

## 使用 python 中的 itertools.chain()方法合并列表

我们还可以使用 itertools.chain()方法在 python 中合并两个或多个列表。为此，我们将列表作为参数传递给 itertools.chain()方法，该方法返回一个 iterable，其中包含要合并的列表的所有元素，这些元素可以进一步转换为一个列表。

```py
import itertools
list1=[1,2,3,4]
list2=[5,6,7,8]
print("First list is:")
print(list1)
print("Second list is:")
print(list2)
merged_list=list(itertools.chain(list1,list2))
print("Merged list is:")
print(merged_list)
```

输出:

```py
First list is:
[1, 2, 3, 4]
Second list is:
[5, 6, 7, 8]
Merged list is:
[1, 2, 3, 4, 5, 6, 7, 8]
```

当我们使用 itertools.chain()方法合并列表时，没有一个输入列表被修改。这种在 python 中合并列表的方式可以方便地合并两个以上的列表，因为我们只需将输入列表作为参数传递给 itertools.chain()方法。

## 结论

在本文中，我们看到了使用不同的方法和模块在 python 中合并两个或多个列表的各种方法。我们还可以使用 [python try except](https://www.pythonforbeginners.com/error-handling/python-try-and-except) 编写本文中使用的程序，并使用异常处理来使程序更加健壮，并以系统的方式处理错误。请继续关注更多内容丰富的文章。