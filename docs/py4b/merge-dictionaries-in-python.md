# 在 Python 中合并字典

> 原文：<https://www.pythonforbeginners.com/dictionary/merge-dictionaries-in-python>

在编程时，可能会出现需要在 python 中合并两个或更多字典的情况。在本文中，我们将研究如何使用不同的方法在 python 中合并两个或更多的字典。

## 使用 items()方法合并字典。

我们可以使用 items()方法通过将一个字典的条目逐个添加到另一个字典来合并两个字典。在字典上调用 items()方法时，会返回包含键值对的元组列表。如果我们必须合并两个字典，我们将一个字典的条目一个接一个地添加到另一个字典中，如下所示。

```py
dict1={1:1,2:4,3:9}
print("First dictionary is:")
print(dict1)
dict2={4:16,5:25,6:36}
print("Second dictionary is:")
print(dict2)
for key,value in dict2.items():
    dict1[key]=value
print("Merged dictionary is:")
print(dict1)
```

输出:

```py
First dictionary is:
{1: 1, 2: 4, 3: 9}
Second dictionary is:
{4: 16, 5: 25, 6: 36}
Merged dictionary is:
{1: 1, 2: 4, 3: 9, 4: 16, 5: 25, 6: 36}
```

在使用上述方法时，我们必须考虑这样一点:如果两个字典之间有公共键，那么最终的字典将具有公共键，这些公共键的值来自我们从中提取条目的第二个字典。此外，只有添加条目的第一个字典会被修改，而第二个字典不会发生任何变化。

## 使用 popitem()方法合并词典。

popitem()方法用于从 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)中删除一个条目。在字典上调用时，popitem()方法删除字典中最近添加的项，并以元组的形式作为键值对返回该项。

为了合并字典，我们将使用 popitem()方法从一个字典中弹出条目，并不断将其添加到另一个字典中。

```py
dict1={1:1,2:4,3:9}
print("First dictionary is:")
print(dict1)
dict2={4:16,5:25,6:36}
print("Second dictionary is:")
print(dict2)
while dict2:
    key,value=dict2.popitem()
    dict1[key]=value
print("Merged dictionary is:")
print(dict1)
```

输出:

```py
First dictionary is:
{1: 1, 2: 4, 3: 9}
Second dictionary is:
{4: 16, 5: 25, 6: 36}
Merged dictionary is:
{1: 1, 2: 4, 3: 9, 6: 36, 5: 25, 4: 16}
```

当使用上述方法时，我们必须记住，要合并的两个字典都被修改。从中删除条目的第二个字典变空，而添加条目的第一个字典也被修改。同样，我们必须考虑这一点，如果在被合并的字典之间有公共键，合并的字典将包含来自第二个字典的值，公共键的项从该字典中弹出。

## 使用 keys()方法。

我们还可以使用 keys()方法在 python 中合并两个字典。在字典上调用 keys()方法时，会返回字典中的键列表。我们将使用 keys()方法从字典中获取所有的键，然后我们可以访问这些键的相关值。之后，我们可以将键值对添加到另一个字典中，如下所示。

```py
dict1={1:1,2:4,3:9}
print("First dictionary is:")
print(dict1)
dict2={4:16,5:25,6:36}
print("Second dictionary is:")
print(dict2)
keylist=dict2.keys()
for key in keylist:
    dict1[key]=dict2[key]
print("Merged dictionary is:")
print(dict1)
```

输出:

```py
First dictionary is:
{1: 1, 2: 4, 3: 9}
Second dictionary is:
{4: 16, 5: 25, 6: 36}
Merged dictionary is:
{1: 1, 2: 4, 3: 9, 4: 16, 5: 25, 6: 36}
```

当我们在 python 中使用上述方法合并两个字典时，我们必须记住，我们从中获取键的第二个字典不会被修改。只有添加了键和值的字典才会被修改。此外，如果被合并的两个字典具有共同的关键字，那么合并的字典将具有来自从中获得关键字的字典的共同关键字的值。

## 使用 update()方法合并字典

我们可以使用 update()方法直接将一个字典合并到另一个字典。在一个字典上调用 update()方法，它将另一个字典作为输入参数。我们可以使用 update()方法合并字典，如下所示。

```py
dict1={1:1,2:4,3:9}
print("First dictionary is:")
print(dict1)
dict2={4:16,5:25,6:36}
print("Second dictionary is:")
print(dict2)
dict1.update(dict2)
print("Merged dictionary is:")
print(dict1)
```

输出:

```py
First dictionary is:
{1: 1, 2: 4, 3: 9}
Second dictionary is:
{4: 16, 5: 25, 6: 36}
Merged dictionary is:
{1: 1, 2: 4, 3: 9, 4: 16, 5: 25, 6: 36}
```

在这种情况下，当我们使用 python 中的 update()方法合并两个字典时，公共键被赋予字典中作为参数传递的值。同样，作为参数传递的第二个字典不会被修改，而调用 update()方法的第一个字典会被修改并转换为最终字典。

## 使用**运算符合并词典

双星号(**)运算符用于将可变长度的关键字参数传递给函数。我们还可以使用**运算符来合并两个字典。当我们对字典应用**操作符时，它反序列化字典并将其转换为键值对的集合，然后使用键值对形成新的字典。

```py
dict1={1:1,2:4,3:9}
print("First dictionary is:")
print(dict1)
dict2={4:16,5:25,6:36}
print("Second dictionary is:")
print(dict2)
mergedDict={**dict1,**dict2}
print("Merged dictionary is:")
print(mergedDict)
```

输出:

```py
First dictionary is:
{1: 1, 2: 4, 3: 9}
Second dictionary is:
{4: 16, 5: 25, 6: 36}
Merged dictionary is:
{1: 1, 2: 4, 3: 9, 4: 16, 5: 25, 6: 36}
```

我们可以使用**运算符一次合并任意数量的字典。此外，不得不合并的原始字典都不会被修改。它们保持原样。在该方法中，当要合并的字典具有公共关键字时，各个关键字的值将从合并时序列中最后出现的字典中取得。

## 结论

在本文中，我们看到了如何使用不同的方法在 python 中合并字典。我们还可以使用 [python try except](https://www.pythonforbeginners.com/error-handling/python-try-and-except) 编写本文中使用的程序，并使用异常处理来使程序更加健壮，并以系统的方式处理错误。请继续关注更多内容丰富的文章。