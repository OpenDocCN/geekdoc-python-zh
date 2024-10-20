# 如何使用 Python divmod()函数

> 原文：<https://www.askpython.com/python/built-in-methods/python-divmod-function>

在本文中，我们将详细介绍 Python divmod()函数的工作和使用。

## 1.Python divmod()函数

[Python](https://www.askpython.com/python) 内置了 divmod()函数，对两个输入值进行**除法**和**模数**运算。

`divmod()`函数以**两个值作为自变量**，并执行除法即 **value1/value2** 和模运算即 **value1%value2** ，**返回商和余数**成对。

**语法:**

```py
divmod(value1, value2)

```

*   **value1** 被视为**提名人**
*   **值 2** 被当作**分母**

**例 1:**

```py
x= 10
y= 2

res_div = divmod(x,y)
print(res_div)

```

在上面的例子中，我们将 10 和 2 传递给了 divmod()函数。此外，divmod()函数执行除法运算，即 **10/2** 和模运算，即 **10%2** ，并从中返回商和余数。

**输出:**

```py
(5, 0) 

```

**例 2:**

```py
x= int(input())
y= int(input())

res_div = divmod(x,y)
print(res_div)

```

**输出:**

```py
(2, 0)

```

**例 3:**

```py
x= 10
y= 3

res_div = divmod(x,y)
print(res_div)

```

**输出:**

```py
(3, 1)

```

* * *

### **带浮点值的 Python divmod()函数**

当 divmod()函数遇到作为参数的**浮点值**时，该函数以如上所示的类似方式计算商和余数。

但是，当一个浮点值作为参数传递给 divmod()函数时，它**返回的商值只考虑所获得的值**的整个部分，即它忽略了小数部分。

**举例:**

```py
x= 10.5
y= 5

res_div = divmod(x,y)
print(res_div)

```

如上所示，divmod(10.5，5)将返回(2.0，0.5)，因为当遇到浮点值时，它会忽略结果中的小数部分，即 10.5/5 将是 2.0 而不是 2.1。因此，省略了小数部分。

**输出:**

```py
(2.0, 0.5)

```

**例 2:**

```py
x= 10
y= 2.4

res_div = divmod(x,y)
print(res_div)

```

**输出:**

```py
(4.0, 0.40000000000000036)

```

* * *

### Python divmod()函数-错误和异常

**1。**如果传递给 divmod()函数的**第一个参数**的值是**零(0)** ，那么该函数返回一对作为 **(0，0)** 。

**例 1:**

```py
x= 0
y= 3

res_div = divmod(x,y)
print(res_div)

```

**输出:**

```py
(0, 0)

```

**2。**如果传递给 divmod()函数的**第二个参数**的值似乎是**零(0)** ，那么该函数返回一个 **ZeroDivisionError，即除以零错误**。

**例 2:**

```py
x= 5
y= 0

res_div = divmod(x,y)
print(res_div)

```

**输出:**

```py
Traceback (most recent call last):
  File "main.py", line 4, in <module>
    res_div = divmod(x,y)
ZeroDivisionError: integer division or modulo by zero

```

**3。**如果 divmod()函数遇到一个**复数**作为参数，它会引发**类型错误**异常

**例 3:**

```py
inp1 =10 + 5J
inp2 = 4
res_div = divmod(inp1, inp2)
print(res_div)

```

**输出:**

```py
Traceback (most recent call last):
  File "main.py", line 4, in <module>
    res_div = divmod(inp1, inp2)
TypeError: can't take floor or mod of complex number.

```

* * *

## 摘要

*   Python divmod()函数接受两个值作为参数列表，并对这两个值执行除法和求模运算。
*   divmod()函数成对返回**商**和**余数**。
*   如果将一个**浮点值**传递给 divmod()函数，该函数将通过**省略相应结果值中的小数部分**来返回一对商和余数。
*   如果传递给 divmod()函数的第二个参数**是零(0)** ，则引发一个 **ZeroDivisionError** 。
*   如果将复数作为参数传递给该函数，该函数将引发 TypeError 异常。

* * *

## 结论

因此，在本文中，我们已经了解了 Python divmod()函数的工作原理。

* * *

## 参考

*   python divmod()–journal ev