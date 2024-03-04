# 如何在 Python 中修改元组

> 原文：<https://www.pythonforbeginners.com/basics/how-to-modify-a-tuple-in-python>

我们知道 tuple 是一种不可变的数据类型，不像 python 字典或列表。这意味着，我们不能以任何方式修改一个元组。但是，我们可能需要修改一个元组。在这种情况下，我们没有其他选择来创建一个包含所需元素的新元组。在本文中，我们将使用不同的方式来修改元组，比如切片、打包和解包。

## 如何将元素追加到元组中

要在元组的末尾追加元素，我们可以使用元组串联，就像我们使用[字符串串联](https://www.pythonforbeginners.com/concatenation/string-concatenation-and-formatting-in-python)将一个字符串追加到另一个字符串。为此，首先，我们将创建一个具有单个元素的元组，该元素必须附加到元组中。然后，我们将使用+运算符连接新元组和现有元组，如下所示。

```py
myTuple = (1, 2, 3, 4)
print("Original Tuple is:", myTuple)
value = 5
print("Value to be added:", value)
newTuple = (5,)
myTuple = myTuple + newTuple
print("Updated tuple is:", myTuple) 
```

输出:

```py
Original Tuple is: (1, 2, 3, 4)
Value to be added: 5
Updated tuple is: (1, 2, 3, 4, 5)
```

我们还可以使用上面的方法将一个元素附加到元组的开头，如下所示。

```py
myTuple = (1, 2, 3, 4)
print("Original Tuple is:", myTuple)
value = 5
print("Value to be added:", value)
newTuple = (5,)
myTuple = newTuple + myTuple
print("Updated tuple is:", myTuple) 
```

输出:

```py
Original Tuple is: (1, 2, 3, 4)
Value to be added: 5
Updated tuple is: (5, 1, 2, 3, 4)
```

为了向元组追加元素，我们还可以使用*操作符进行打包和解包。首先，我们将使用*操作符解包现有的元组。之后，我们将创建一个包含新元素的新元组，如下所示。

```py
myTuple = (1, 2, 3, 4)
print("Original Tuple is:", myTuple)
value = 5
print("Value to be added:", value)
myTuple = (*myTuple, value)
print("Updated tuple is:", myTuple) 
```

输出:

```py
Original Tuple is: (1, 2, 3, 4)
Value to be added: 5
Updated tuple is: (1, 2, 3, 4, 5)
```

就像将元素追加到元组的末尾一样，我们可以使用如下的打包和解包将元素追加到元组的开头。

```py
myTuple = (1, 2, 3, 4)
print("Original Tuple is:", myTuple)
value = 5
print("Value to be added:", value)
myTuple = (value, *myTuple)
print("Updated tuple is:", myTuple) 
```

输出:

```py
Original Tuple is: (1, 2, 3, 4)
Value to be added: 5
Updated tuple is: (5, 1, 2, 3, 4)
```

## 如何在元组中的特定位置插入元素

要在元组中的特定位置插入元素，我们可以使用切片。如果我们必须在元组的索引“I”处插入一个元素，我们将对元组进行切片并创建两个新的元组。第一元组将包含从原始元组的索引 0 到 i-1 的元素。第二元组将包含从索引“I”到最后的元素。之后，我们将创建一个元组，其中包含一个必须插入到索引 I 处的元组中的元素。然后，我们将连接这三个元组，以便在索引“I”处插入新元素，如下所示。

```py
myTuple = (1, 2, 3, 4, 5, 6, 7, 8)
print("Original Tuple is:", myTuple)
value = 0
print("Value to be inserted at index 3:", value)
left_tuple = myTuple[0:3]
right_tuple = myTuple[3:]

myTuple = left_tuple + (value,) + right_tuple
print("Updated tuple is:", myTuple) 
```

输出:

```py
Original Tuple is: (1, 2, 3, 4, 5, 6, 7, 8)
Value to be inserted at index 3: 0
Updated tuple is: (1, 2, 3, 0, 4, 5, 6, 7, 8)
```

我们还可以使用打包和解包来合并新创建的元组，如下所示。

```py
myTuple = (1, 2, 3, 4, 5, 6, 7, 8)
print("Original Tuple is:", myTuple)
value = 0
print("Value to be inserted at index 3:", value)
left_tuple = myTuple[0:3]
right_tuple = myTuple[3:]

myTuple = (*left_tuple, value, *right_tuple)
print("Updated tuple is:", myTuple) 
```

输出:

```py
Original Tuple is: (1, 2, 3, 4, 5, 6, 7, 8)
Value to be inserted at index 3: 0
Updated tuple is: (1, 2, 3, 0, 4, 5, 6, 7, 8)
```

## 如何通过删除元素来修改元组

为了从元组的开头删除一个元素，我们将使用剩余的元素创建一个新的元组，如下所示。

```py
myTuple = (1, 2, 3, 4, 5, 6, 7, 8)
print("Original Tuple is:", myTuple)
myTuple = myTuple[1:]
print("Updated tuple is:", myTuple)
```

输出:

```py
Original Tuple is: (1, 2, 3, 4, 5, 6, 7, 8)
Updated tuple is: (2, 3, 4, 5, 6, 7, 8) 
```

如果我们需要删除一个元组的最后一个元素，我们将按如下方式切分剩余的元素。

```py
myTuple = (1, 2, 3, 4, 5, 6, 7, 8)
print("Original Tuple is:", myTuple)
tupleLength = len(myTuple)
myTuple = myTuple[0:tupleLength - 1]
print("Updated tuple is:", myTuple)
```

输出:

```py
Original Tuple is: (1, 2, 3, 4, 5, 6, 7, 8)
Updated tuple is: (1, 2, 3, 4, 5, 6, 7)
```

为了删除存在于索引“I”处的元素，我们将创建原始元组的两个片段。第一个片将包含原始元组的从索引 0 到 i-1 的元素。第二元组将包含从索引“i+1”到最后的元素。之后，我们将连接新创建的元组，排除索引“I”处的元素，如下所示。

```py
myTuple = (1, 2, 3, 4, 5, 6, 7, 8)
print("Original Tuple is:", myTuple)
left_tuple = myTuple[0:3]
right_tuple = myTuple[4:]
myTuple = left_tuple + right_tuple
print("Updated tuple after deleting element at index 3 is:", myTuple) 
```

输出:

```py
Original Tuple is: (1, 2, 3, 4, 5, 6, 7, 8)
Updated tuple after deleting element at index 3 is: (1, 2, 3, 5, 6, 7, 8)
```

## 如何修改元组中特定索引处的元素

为了修改元组索引“I”处的元素，我们将创建原始元组的两个切片。第一个片将包含原始元组的从索引 0 到 i-1 的元素。第二个切片将包含从索引“i+1”到最后的元素。之后，我们将通过在如下位置插入新值来更新索引“I”处的元素。

```py
myTuple = (1, 2, 3, 4, 5, 6, 7, 8)
print("Original Tuple is:", myTuple)
left_tuple = myTuple[0:3]
right_tuple = myTuple[4:]
myTuple = left_tuple + (100,) + right_tuple
print("Updated tuple after modifying element at index 3 is:", myTuple) 
```

输出:

```py
Original Tuple is: (1, 2, 3, 4, 5, 6, 7, 8)
Updated tuple after modifying element at index 3 is: (1, 2, 3, 100, 5, 6, 7, 8)
```

我们还可以使用*运算符来合并新创建的元组，如下所示。

```py
myTuple = (1, 2, 3, 4, 5, 6, 7, 8)
print("Original Tuple is:", myTuple)
left_tuple = myTuple[0:3]
right_tuple = myTuple[4:]
myTuple = (*left_tuple, 100, *right_tuple)
print("Updated tuple after modifying element at index 3 is:", myTuple) 
```

输出:

```py
Original Tuple is: (1, 2, 3, 4, 5, 6, 7, 8)
Updated tuple after modifying element at index 3 is: (1, 2, 3, 100, 5, 6, 7, 8)
```

## 结论

在本文中，我们使用了不同的方法来修改 python 中的元组。要阅读更多关于 python 编程的内容，请阅读这篇关于[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。请继续关注更多内容丰富的文章。