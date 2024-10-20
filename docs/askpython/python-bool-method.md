# Python bool()方法:您想知道的一切

> 原文：<https://www.askpython.com/python/built-in-methods/python-bool-method>

你好。今天在本教程中，我们将学习 Python bool()方法。

所以，让我们进入正题。

## Python bool()方法

bool()方法是一个内置的 Python 方法，它将标准的真值测试过程应用于传递的**对象/值**，并返回一个[布尔值](https://docs.python.org/3.7/library/stdtypes.html#bltin-boolean-values)。此外，bool 类不能被子类化。它仅有的实例是`False`和`True`。下面给出了使用该方法的语法。

```py
bool([x])

```

这里， `x` 是一个可选参数，可以是一个对象、某个值或者任何表达式。当 True 被传递时，该方法返回`True`,同样，当 False 被传递时，该方法返回`False`。

`bool()`方法为下述条件返回`False`。否则，它返回`True`。

*   如果对象有一个已定义的 **__bool__()** 方法，那么布尔结果取决于它返回的内容。否则，如果对象定义了 **__len__()** ，而不是 **__bool__()** ，则考虑其返回值。
*   如果值为**零**任何类型(0，0.0，0j 等。),
*   如果对象是一个**空**集合或序列，如[列表](https://www.askpython.com/python/list/python-list)、 [st](https://www.askpython.com/python/string) [r](https://www.askpython.com/python/string) [ing](https://www.askpython.com/python/string) 、[元组](https://www.askpython.com/python/tuple/python-tuple)、[字典](https://www.askpython.com/python/dictionary/python-dictionary-dict-tutorial)等。
*   如果值是`False`或`None`常数。

**注意:**如果对于任何一个对象`x`，没有定义 `__bool__()`或者`__len__()`方法，我们得到的结果是`True`。

## Python bool()工作

现在我们已经学习了`bool()`方法的基础，让我们尝试一些例子来更好地理解。

### 1.用数字

下面的例子说明了使用数字的`bool()`方法。

```py
from fractions import Fraction
from decimal import Decimal

# variable
val = 55
val1 = 0
print("bool(val):", bool(val))
print("bool(val1):", bool(val1))

# expression
print("bool(20 - 4):", bool(20 - 4))
print("bool(20 - 20):", bool(20 - 20))

# float
print("bool(0.0001):", bool(0.0001))
print("bool(0.00):", bool(0.00))

# hex
print("bool(0xF):", bool(0xF))

# Complex Numbers
print("bool(12 - 34j):", bool(12 - 34j))
print("bool(0j):", bool(0j))

# decimal floating point and fractional numbers
print("bool(Decimal(0)):", bool(Decimal(0)))
print("bool(Fraction(0, 2)):", bool(Fraction(0, 2)))

```

**输出:**

```py
bool(val): True
bool(val1): False
bool(20 - 4): True
bool(20 - 20): False
bool(0.0001): True
bool(0.00): False
bool(0xF): True
bool(12 - 34j): True
bool(0j): False
bool(Decimal(0)): False
bool(Fraction(0, 2)): False

```

### 2.用绳子

对于字符串，`bool()`方法返回 True，直到它的`len()`等于**零** (0)。

```py
# bool() with strings

string1 = "Python"  # len > 0
print(bool(string1))

string1 = ''  # len = 0
print(bool(string1))

string1 = 'False'  # len > 0
print(bool(string1))

string1 = '0'  # len > 0
print(bool(string1))

```

**输出:**

```py
True
False
True
True

```

### 3.带有内置对象

对于序列或集合，只有当传递的对象是空的**时，该方法才返回`False`。**

```py
# list
a = [1, 2, 3, 4]
print(bool(a))

# empty objects
a = [] 
print(bool(a))

a = ()
print(bool(a))

a = {}
print(bool(a)) 
```

****输出:****

```py
True
False
False
False 
```

### **4.使用自定义对象**

**在下面的例子中，我们为我们的`custom`类定义了`__init__()`和`__bool__()`方法。我们用不同的值构造两个对象 **x** 和 **y** 。**

****注意:**即使我们为自定义类定义了`__len__()`，也不会影响`bool()`的结果，因为我们已经定义了`__bool__()`。_ _ len _ _()的返回值仅在类没有定义 __bool__()时才考虑。**

```py
class custom():
    val = 0
    def __init__(self, num):
        self.val = num 
    def __bool__(self):
        return bool(self.val)

# custom objects
x = custom(0)
y = custom(52)

print(bool(x))
print(bool(y)) 
```

****输出:****

```py
False
True 
```

**这里，定制对象`x`和`y`的`bool()`结果间接依赖于传递的参数(x 的 **0** ，y 的 **52** )。因此，我们得到 x 的`False`(bool(0)= False)和 y 的`True`(bool(52)= True)。**

## **包扎**

**今天到此为止。希望你已经清楚地理解了这个主题 Python 中的 **bool()方法**。我们建议浏览参考资料一节中提到的链接，以获得关于该主题的更多信息。**

**如有任何进一步的问题，欢迎使用下面的评论。**

## **参考**

*   **[Python bool()](https://docs.python.org/3.7/library/functions.html#bool) 文件，**
*   **[Python bool 类型](https://docs.python.org/3.7/library/stdtypes.html#bltin-boolean-values)–文档，**
*   **[bool()在 Python 中的实际应用是什么？](https://stackoverflow.com/questions/24868086/what-is-the-practical-application-of-bool-in-python)–堆栈溢出问题。**