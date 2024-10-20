# 用 Python 复制字典的 4 种简单方法

> 原文：<https://www.askpython.com/python/dictionary/copy-a-dictionary-in-python>

## 介绍

在本教程中，我们将讨论用 Python 复制字典的各种方法或技术。

理论上，Python 中的[字典](https://www.askpython.com/python/dictionary/python-dictionary-dict-tutorial)是数据值的无序集合，它将相应的元素存储为**键-项**对。此外，用户可以使用相应的**键**来访问每个项目。

那么，让我们直接进入复制程序。

## 用 Python 复制字典的方法

在这一节中，我们将详细阐述用 Python 复制字典的不同方法。让我们一个一个地检查它们。

### 1.逐个元素复制字典

在这种技术中，我们遍历整个字典，并将键指向的每个元素复制到一个先前声明的新字典中。看看下面的代码:

```py
#given dictionary
dict1={0:'1',1:'2',3:[1,2,3]}
print("Given Dictionary:",dict1)
#new dictionary
dict2={}
for i in dict1:
    dict2[i]=dict1[i] #element by elemnet copying

print("New copy:",dict2)
#Updating dict2 elements and checking the change in dict1
dict2[1]=33
dict2[3][1]='22' #list item updated
print("Updated copy:",dict2)
print("Given Dictionary:",dict1)

```

**输出**:

```py
Given Dictionary: {0: '1', 1: '2', 3: [1, 2, 3]}
New copy: {0: '1', 1: '2', 3: [1, 2, 3]}

Updated copy: {0: '1', 1: 33, 3: [1, '22', 3]}
Given Dictionary: {0: '1', 1: '2', 3: [1, '22', 3]}

```

这里，

*   我们初始化了一个字典， **dict1**
*   在我们打印它之后，我们声明一个空字典， **dict2** 我们将在那里复制 dict1
*   接下来，我们使用循环的[遍历 dict1。并且使用操作`dict2[i]=dict1[i]`，我们将每个元素从 **dict1** 复制到 **dict2** 。](https://www.askpython.com/python/python-for-loop)

现在我们的复制已经完成，记住`=`操作符在字典中为 iterable 对象创建引用。因此，如果更新了 **dict2** 中的**不可迭代**元素，那么 **dict1** 中的相应元素将保持不变。

然而，如果像列表项这样的**可迭代**对象被改变，我们也会看到**字典 1** 列表的改变。上面代码的第二部分解释了这一点。尝试在更新后比较 dict1 和 dict2 的结果。我们看到上面的陈述是真实的。

### 2.使用=运算符在 Python 中复制字典

让我们看看如何使用单个 **'='** 操作符直接复制 Python 中的字典。

```py
#given dictionary
dict1={1:'a',2:'b',3:[11,22,33]}
print("Given Dictionary:",dict1)
#new dictionary
dict2=dict1 #copying using = operator
print("New copy:",dict2)

#Updating dict2 elements and checking the change in dict1
dict2[1]=33
dict2[3][2]='44' #list item updated

print("Updated copy:",dict2)
print("Given Dictionary:",dict1)

```

**输出**:

```py
Given Dictionary: {1: 'a', 2: 'b', 3: [11, 22, 33]}

New copy: {1: 'a', 2: 'b', 3: [11, 22, 33]}

Updated copy: {1: 33, 2: 'b', 3: [11, 22, '44']}
Given Dictionary {1: 33, 2: 'b', 3: [11, 22, '44']}

```

在上面的代码中，

*   首先我们初始化一个字典， **dict1** 。并通过代码行`dict2=dict1`直接复制到新对象 **dict2**
*   该操作将 dict1 中存在的每个对象的引用复制到新字典 dict2 中
*   因此，更新 dict2 的任何元素都会导致 dict1 发生变化，反之亦然。
*   从上面的代码可以清楚地看出，当我们更新 **dict2** 中的任何(可迭代或不可迭代)对象时，我们也会在 **dict1** 中看到相同的变化。

### 3.使用 copy()方法

Python 中字典的`copy()`方法返回给定字典的**浅拷贝**。这类似于我们之前看到的通过遍历字典来复制元素的情况。

也就是说，字典元素的引用被插入到新字典中(浅层拷贝)。看看下面的代码:

```py
#given dictionary
dict1={ 10:'a', 20:[1,2,3], 30: 'c'}
print("Given Dictionary:",dict1)
#new dictionary
dict2=dict1.copy() #copying using copy() method
print("New copy:",dict2)

#Updating dict2 elements and checking the change in dict1
dict2[10]=10
dict2[20][2]='45' #list item updated

print("Updated copy:",dict2)
print("Given Dictionary:",dict1)

```

**输出**:

```py
Given Dictionary: {10: 'a', 20: [1, 2, 3], 30: 'c'}
New copy: {10: 'a', 20: [1, 2, 3], 30: 'c'}

Updated copy: {10: 10, 20: [1, 2, '45'], 30: 'c'}
Given Dictionary: {10: 'a', 20: [1, 2, '45'], 30: 'c'}

```

在上面的代码中:

*   我们用一些值初始化一个字典 **dict1** 。并在其上使用`copy()`方法创建一个浅层副本
*   复制完成后，我们更新新元素，并在原始字典中看到相应的变化
*   类似于**逐元素**复制技术的情况，这里 **dict2** 的不可迭代元素的改变对原始字典没有任何影响
*   而对于像列表这样的可迭代的字典，这种变化也反映在给定的字典中， **dict1**

### 4.使用 copy.deepcopy()方法在 Python 中复制字典

Python 中的`deepcopy()`方法是**复制**模块的成员。它返回一个新的字典，其中包含所传递字典的复制元素。注意，这个方法以一种**递归**的方式复制给定字典的所有元素。让我们看看如何使用它，

```py
import copy

#given dictionary
dict1={ 10:'a', 20:[1,2,3], 30: 'c'}
print("Given Dictionary:",dict1)
#new dictionary
dict2=copy.deepcopy(dict1) #copying using deepcopy() method

print("New copy:",dict2)
#Updating dict2 elements and checking the change in dict1
dict2[10]=10
dict2[20][2]='45' #list item updated

print("Updated copy:",dict2)
print("Given Dictionary:",dict1)

```

**输出**:

```py
Given Dictionary: {10: 'a', 20: [1, 2, 3], 30: 'c'}
New copy: {10: 'a', 20: [1, 2, 3], 30: 'c'}

Updated copy: {10: 10, 20: [1, 2, '45'], 30: 'c'}
Given Dictionary: {10: 'a', 20: [1, 2, 3], 30: 'c'}

```

现在，

*   在第一行中，我们初始化原始字典 **dict1** ，
*   我们使用`copy.deepcopy()`方法复制新字典中的 dict1 元素， **dict2** ，
*   成功复制后，我们更新新的副本并查看原始字典中的变化，
*   从输出中我们可以看到， **dict2** 的任何变化都是**而不是**反映在 **dict1** 中。因此，当我们需要在代码中更改新的字典，同时保持原来的字典不变时，这种方法非常有用。

## 结论

所以，在本教程中，我们学习了用 Python 中的 **4** 不同的**复制字典的方法。关于这个话题的任何问题，请随意使用下面的评论。**

## 参考

*   [浅层和深层复制操作](https://docs.python.org/3/library/copy.html)–Python 复制文档，
*   [如何复制一本字典并且只编辑副本](https://stackoverflow.com/questions/2465921/how-to-copy-a-dictionary-and-only-edit-the-copy)–stack overflow 问题，
*   [快速复制字典的方法](https://stackoverflow.com/questions/5861498/fast-way-to-copy-dictionary-in-python)–stack overflow 问题。