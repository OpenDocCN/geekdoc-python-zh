# Python eval()函数

> 原文：<https://www.askpython.com/python/built-in-methods/python-eval>

读者们，你们好。在本文中，我们将重点关注 **Python eval()函数**。

## 了解 Python eval()函数

Python eval()函数转换并计算传递给它的表达式。

注意:该方法仅用于测试目的。eval()函数不整理传递给它的表达式。如果恶意用户在这里执行 Python 代码，它很容易成为您的服务器的漏洞。

`eval() function`解析 python 表达式，并在 python 程序中运行作为参数传递给它的代码。

**语法:**

```py
eval('expression', globals=None, locals=None)

```

*   `expression`:可以是用户想要在 python 代码本身内计算的任何 python 表达式(字符串参数)。
*   `globals`:这是一个[字典](https://www.askpython.com/python/dictionary/python-dictionary-dict-tutorial)，指定了 eval()函数可以执行的表达式。
*   `locals`:描述 eval()函数可以利用的局部方法和数据变量。

**例 1:** **传递表达式添加两个局部变量**

```py
a=20
b=30
res = eval('a+b')
print(res)

```

在上面的代码片段中，我们向 eval()函数传递了一个表达式“a+b ”,以便添加两个局部变量:a 和 b。

**输出:**

```py
50

```

**例 2:** **用户输入的 Python eval()函数**

```py
num1 = int(input())
num2 = int(input())
mult = eval('num1*num2')
print('Multiplication:',mult)

```

在上面的例子中，我们接受了用户的输入，并将其赋给变量。此外，我们已经传递了这两个输入值相乘的表达式。

**输出:**

```py
30
20
Multiplication: 600

```

* * *

## 带有熊猫模块的 Python eval()函数

Python eval 函数也可以和 [Pandas 模块](https://www.askpython.com/python-modules/pandas/python-pandas-module-tutorial)一起操作。pandas.eval()函数接受表达式，并在 python 程序中执行同样的操作。

**语法:**

```py
DataFrame.eval('expression',inplace)

```

*   `expression`:包含在字符串引号中的 python 表达式，将在 python 程序中执行。
*   `inplace`:默认值=真。如果 python 表达式被证明是赋值表达式，则 inplace 决定执行该操作并改变数据框对象。如果为 FALSE，则会创建一个新的 dataframe 并作为结果返回。

**示例 1:传递 inplace = TRUE 的表达式**

```py
import pandas
data = pandas.DataFrame({"x":[1,2,3,4],"y":[1,3,5,7],"w":[2,4,6,8],"z":[1,1,1,1]})
print("Original data:\n")
print(data)
data.eval('z = x * y', inplace = True)
print("Data after executing eval():\n")
print(data)

```

在上面的示例中，我们创建了一个数据帧，并传递了一个要在 python 脚本中执行的表达式。

当 inplace 设置为 TRUE 时，从表达式中获得的数据值将存储在同一个 dataframe 对象“data”中。

**输出:**

```py
 Original data:

   x  y  w  z
0  1  1  2  1
1  2  3  4  1
2  3  5  6  1
3  4  7  8  1

Data after executing eval():

   x  y  w   z
0  1  1  2   1
1  2  3  4   6
2  3  5  6  15
3  4  7  8  28

```

**示例 2:使用 inplace = FALSE 在 python 脚本中执行表达式**

```py
import pandas
data = pandas.DataFrame({"x":[1,2,3,4],"y":[1,3,5,7],"w":[2,4,6,8],"z":[1,1,1,1]})
print("Original data:\n")
print(data)
data1 = data.eval('z = x * y', inplace = False)
print("Data after executing eval():\n")
print(data1)

```

在上面的代码片段中，我们将 inplace = FALSE 传递给了 eval()函数。因此，python 表达式的结果将存储在新的 dataframe 对象“data1”中。

**输出:**

```py
Original data:

   x  y  w  z
0  1  1  2  1
1  2  3  4  1
2  3  5  6  1
3  4  7  8  1
Data after executing eval():

   x  y  w   z
0  1  1  2   1
1  2  3  4   6
2  3  5  6  15
3  4  7  8  28

```

* * *

## eval()函数的安全问题

*   Python eval()函数更容易受到安全威胁。
*   通过 Python eval()函数可以很容易地获得敏感的用户输入数据。
*   因此，提供了参数 globals 和 locals 来限制对数据的直接访问。

* * *

## 摘要

*   Python eval()函数用于直接在 Python 脚本中**执行 python 表达式。**
*   eval()函数也可以以类似的方式用于 **Pandas 模块**。
*   Python eval()函数更容易受到**安全威胁**。因此，在执行 eval()函数之前，有必要检查传递给它的信息。

* * *

## 结论

因此，在本文中，我们已经了解了 Python eval()函数的工作原理和漏洞。

* * *

## 参考

*   Python eval()函数— JournalDev