# 用 Python 创建字典的方法

> 原文：<https://www.pythonforbeginners.com/dictionary/ways-to-create-dictionary-in-python>

在 python 中，字典是一种数据结构，我们可以在其中以键值对的形式保存数据。在本文中，我们将研究和实现用 python 创建字典的各种方法。

## 用 python 创建一个空字典

我们可以使用花括号或字典构造函数创建一个空的 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)。

要使用花括号创建一个空字典，我们只需声明一个变量并分配一对花括号，如下所示。

```py
myDict={}
print("Created dictionary is:")
print(myDict)
```

输出:

```py
Created dictionary is:
{}
```

在上面的输出中，花括号显示已经创建了一个空字典。

要使用字典构造函数创建一个空字典，我们可以简单地调用构造函数`dict()`并将其赋给一个变量，如下所示。

```py
myDict=dict()
print("Created dictionary is:")
print(myDict)
```

输出:

```py
Created dictionary is:
{}
```

在上面的程序中，我们已经使用`dict()`构造函数创建了一个空字典。我们还可以使用上面的方法来创建已经定义了键值对的字典，这将在下面的章节中讨论。

## 用已经定义的键值对创建字典

要使用已经定义的键值对创建字典，我们可以使用花括号、字典构造函数或字典理解。

要使用花括号创建预定义键值对的字典，我们只需使用冒号编写键值对，并使用逗号分隔每个键值对，如下所示。

```py
myDict={"name":"PythonForBeginners","acronym":"PFB"}
print("Created dictionary is:")
print(myDict)
```

输出:

```py
Created dictionary is:
{'name': 'PythonForBeginners', 'acronym': 'PFB'}
```

在上面的程序中，我们创建了一个字典，将`name`和`acronym`作为关键字，将`PythonForBeginners`和`PFB`分别作为字典中关键字的值。我们可以使用字典构造函数执行相同的操作。

在使用字典构造函数创建具有预定义键值对的字典时，我们将键和值作为关键字参数传递给 python 字典构造函数，如下所示。

```py
myDict=dict(name="PythonForBeginners",acronym="PFB")
print("Created dictionary is:")
print(myDict)
```

输出:

```py
Created dictionary is:
{'name': 'PythonForBeginners', 'acronym': 'PFB'}
```

在输出中，我们可以看到我们已经创建了一个与前面方法相同的字典，但是使用了`dict()`构造函数来实现相同的目的。在程序中，键值对已作为关键字参数传递给构造函数，其中字典的键作为参数的关键字传递，字典的键的相应值作为值传递给关键字参数。

当给定包含键和值对的元组时，我们也可以创建字典。在这种情况下，元组索引 0 处的值成为字典的键，元组索引 1 处的值成为字典中相应键的值。

当我们在一个列表中有键-值对时，这个列表中的元组包含每个元组中的键和值，我们可以使用字典构造函数创建一个字典，如下所示。

```py
myList=[("name","PythonForBeginners"),("acronym","PFB")]
myDict=dict(myList)
print("Created dictionary is:")
print(myDict)
```

输出:

```py
Created dictionary is:
{'name': 'PythonForBeginners', 'acronym': 'PFB'}
```

在上面的程序中，我们可以验证字典的键是从元组中索引 0 处的元素创建的，并且各个键的值是从元组中索引 1 处的元素生成的。

可能会给我们一个键列表和一个单独的值列表来包含在字典中。为了创建字典，我们首先需要创建键值对元组，然后我们可以使用这些键值对创建字典。为了创建包含键和值对的元组，我们可以使用`zip()`函数。

`zip()`函数接受两个 iterable，并执行从第一个 iterable 的元素到第二个 iterable 的元素的一对一映射(基于索引)。如果在任何 iterable 中有额外的元素，它们将被忽略。

我们可以通过使用`zip()`函数和字典构造函数，使用给定的键和值列表创建一个 python 字典，如下所示。

```py
keyList=["name","acronym"]
valueList=["Pythonforbeginners","PFB"]
temp=zip(keyList,valueList)
print("given list of keys is:")
print(keyList)
print("given list of values is:")
print(valueList)
print("Created key value pairs are:")
myList=list(temp)
print(myList)
myDict=dict(myList)
print("Created dictionary is:")
print(myDict)
```

输出:

```py
given list of keys is:
['name', 'acronym']
given list of values is:
['Pythonforbeginners', 'PFB']
Created key value pairs are:
[('name', 'Pythonforbeginners'), ('acronym', 'PFB')]
Created dictionary is:
{'name': 'Pythonforbeginners', 'acronym': 'PFB'}
```

在上面的程序中，首先使用`zip()`函数生成一个包含键值对元组的列表，然后我们使用字典构造函数从元组列表中创建一个字典。

## 使用字典理解创建字典

正如我们使用 [list comprehension](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python) 在 python 中创建列表一样，我们也可以使用 dictionary comprehension 在 python 中创建字典。通过从给定的 iterable 或 dictionary 为新字典创建键值对，我们可以从另一个字典或其他 iterable 创建字典。

假设我们想创建一个字典，其中包含一个列表中的元素作为键，它们的平方作为值，我们可以这样做。

```py
myList=[1,2,3,4]
myDict={x:x**2 for x in myList}
print("List is:")
print(myList)
print("Created dictionary is:")
print(myDict)
```

输出:

```py
List is:
[1, 2, 3, 4]
Created dictionary is:
{1: 1, 2: 4, 3: 9, 4: 16}
```

在输出中，我们可以看到创建的字典包含了作为键的列表元素和作为字典键的各自值的元素的平方。

我们还可以通过基于特定条件从另一个字典中选择键值对来创建一个新字典。对于这个任务，我们可以使用`items()`方法获得作为元组的键值对列表，然后我们可以根据任何给定的条件使用具有键值对的元组来创建新的字典。

假设我们想要创建一个新的字典，在这个字典中只选择原始字典中那些具有偶数作为键的键值对。这可以如下进行。

```py
myDict={1:1,2:2,3:3,4:4}
print("Original Dictionary is")
print(myDict)
newDict={key:value**2 for (key,value) in myDict.items() if key%2==0}
print("New dictionary is:")
print(newDict)
```

输出:

```py
Original Dictionary is
{1: 1, 2: 2, 3: 3, 4: 4}
New dictionary is:
{2: 4, 4: 16}
```

在上面的输出中，我们可以看到，我们已经从原始字典中有选择地为新字典选择了键值对，新字典只包含偶数的键。我们还通过平方修改了各自的值。

## 结论

在本文中，我们看到了用 python 创建字典的几种方法。我们已经看到了如何通过在 python 中实现不同的程序，使用花括号、字典构造器和字典理解来创建字典。请继续关注更多内容丰富的文章。