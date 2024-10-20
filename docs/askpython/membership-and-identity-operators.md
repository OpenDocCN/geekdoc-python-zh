# Python 成员和标识运算符

> 原文：<https://www.askpython.com/python/examples/membership-and-identity-operators>

读者朋友们，你们好！在本文中，我们将详细关注 **Python 成员和身份操作符**。

所以，让我们开始吧！！🙂

* * *

## Python 成员和身份操作符–快速概述！

Python 为我们提供了各种各样的[操作符](https://www.askpython.com/python/python-operators)，以便在更大的范围内对数据值和变量执行操作。在本文的上下文中，我们将主要关注 Python 中两种重要的操作符类型:

1.  **隶属运算符**
2.  **身份运算符**

现在，让我们在下一节中继续了解它们各自的功能。

* * *

## Python 成员运算符–[' in '，' not in']

Python 成员操作符帮助我们评估和验证数据结构(如列表、元组等)中特定序列的数据值的成员关系。我们的意思是，它检查给定的数据序列是否存在于另一个序列或结构中，并对其进行验证。

### 1。python“in”成员运算符

in 运算符是一个布尔运算符，它检查数据结构中是否存在特定的数据序列，如果找到，则返回 true。否则，它返回 false。

**举例:**

在这个例子中，我们在列表(list1)中搜索数据序列值(10，20)。找到后，它打印结果语句。

```py
lst1=[10,20,30,40,50]
lst2=[10,6,20,7]
for x in lst1:
	if x in lst2:
		print("Data overlaps for value:", x)	

```

**输出:**

```py
Data overlaps for value: 10
Data overlaps for value: 20

```

* * *

### 2。Python“不在”成员运算符

如果在列表、[字符串](https://www.askpython.com/python/string/strings-in-python)等序列中没有遇到给定的数据值，则 [not in 运算符](https://www.askpython.com/python/examples/in-and-not-in-operators-in-python)的结果为真。

**举例:**

在本例中，数据值“32”不在列表中，因此它返回 false 并在 if 条件后打印语句。

```py
lst=[10,20,30,40,50]
data = 32
if data not in lst:
   print("Data not found")
else:
   print("Data is present")

```

**输出:**

```py
Data not found

```

* * *

## Python 标识运算符–['是'，'不是']

Python 中的 Identity 操作符帮助我们检查值的相等性，比如它们指向什么样的内存位置，是否具有预期的相同数据类型，等等。

### 1。python“is”标识运算符

使用“is”操作符，我们可以很容易地检查任意一端的值的有效性，无论它们是指向同一个内存点，还是具有相同的数据类型或所需的数据类型，等等。

**举例**:

在下面的例子中，我们使用 is 运算符来检查数据值是否是 float 类型。如果条件满足，则返回 TRUE，否则返回 false。

```py
data = 40.03
if type(data) is float:
	print("TRUE")
else:
	print("FALSE")

```

**输出:**

```py
TRUE

```

* * *

### 2。Python“不是”标识运算符

使用“is not”操作符，我们根据等式或上述条件检查有效性，如果它们不满足，则返回 TRUE。如果条件满足，则返回 FALSE。

```py
data = 40.03
if type(data) is not int:
	print("Not same")
else:
	print("same")

```

**输出:**

```py
Not same

```

* * *

## 结论

如果你遇到任何问题，欢迎在下面评论。更多与 Python 编程相关的帖子，请继续关注我们。在那之前，学习愉快！！🙂