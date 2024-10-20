# Python 中“is”和“==”的区别

> 原文：<https://www.askpython.com/python/examples/difference-between-is-and-double-equals>

嘿伙计们！，我们都知道 Python 有一些大多数解释语言没有的好处。其中一些是我们在进行数学计算时使用的灵活性概念。这样做的时候，我们会对操作符和关键字产生一些混淆。现在，在这篇文章中，我们的目的是得到相同的区别。**是**和**的双等号**运算符。那么，让我们开始吧。

## Python 中的运算符有哪些？

任何编程语言中的运算符都是基础背后的主要基本概念。**Python 中有以下[运算符:](https://www.askpython.com/course/python-course-operators)**

1.  **逻辑:执行逻辑计算**
    1.  和
    2.  或者
    3.  不
2.  **算术:进行基本的算术计算**
    1.  **+** :加法
    2.  **–**:乘法
    3.  **/** :除法
    4.  **%** :取模(返回余数)
    5.  **/**:地板除法运算符(返回浮点除法的整数值)
3.  **指数:计算数的幂和异或值**
    1.  ****** :电源
    2.  **^** :异或

## 让我们对这些概念进行编码，并追踪它们之间的区别

现在，我们将比较这些关键字，并尝试跟踪它们在操作上的差异。

### **"=="** 运算符

让我们以两个变量为例。每个变量都有不同的值。假设 a 有 20，b 有 30。现在，我们都知道他们是不平等的。但是，计算机如何识别这一点呢？为此，我们只使用了 **double equal-to** 运算符。让我们举一个代码例子:

**代码:**

```py
a = 30
b = 30
if a == b:
    print(True) 
else:
    print(False)

```

**输出:**

```py
True

```

这里我们可以说明，等于的主要作用是检查值是否相同。对于一个复杂的例子，我们也可以用函数来检查:

**代码检查一个数的[平方根](https://www.askpython.com/python/examples/calculate-square-root) :**

```py
from math import sqrt
def square_root(val):
    return (val**(1/2))

a = sqrt(9)
b = square_root(9)
print(a == b)

```

**输出:**

```py
True

```

这里我们比较两个函数返回的值。它们返回的值是相同的，但是在解释器中两者的标识是不同的。

```py
print(id(a))
print(id(b))

```

**输出:**

```py
2178644181968
2178644086384

```

这里的双重等于运算符显示了不同的性质。当我们比较它们的身份时，它将返回 false。它返回 false，因此我们使用“is”操作符来解决这个问题。

### **是**关键字

该关键字用于值的比较以及对象引用。当我们创建任何类的对象时，它是保存每个类的属性的实例。对于每一个新创建的对象，解释器都给它们分配一个新的身份。

#### 数字示例

```py
a = 10
b = 10.0
print(a is b)

```

**输出:**

```py
False

```

正如我们所见，两者都是 10，但一个是浮点数，另一个是整数。所以，让这些值看起来相似，对象类型可以不同

#### 数据结构示例

在这个例子中，我们有两个列表 a 和 b。它们都包含相同的元素值。但是，当我们尝试使用关键字 **"is"** 来比较它们时，结果令人惊讶。

```py
a = [1, 2, 3, 4, 5]
b = [1, 2, 3, 4, 5]

print(type(a))
print(type(b))

if (a is b):
    print("Successful")

else:
    print("Unsuccessful")

```

**输出:**

```py
<class 'list'>
<class 'list'>
False

```

**说明:**

1.  两个列表都与`<class>`具有相同的类别。
2.  但是，主要问题是分配给 l1 的存储块不同于 l1 的存储块。
3.  **是**操作符，检查我们创建的对象的内存位置。
4.  每个对象的内存块分配是不同的。这使得**为**返回**假**值。

#### 例如使用 NumPy 阵列

**代码:**

```py
import numpy as np
a = np.array([[1, 2], [3, 4]])
b = np.array([[1, 2], [3, 4]])
print(a is b)

```

**输出:**

```py
False

```

NumPy 数组也是如此。两个阵列是相同的，但是分配给它们的内存是不同的。他们的 **id** 不一样。所以，我们得到**假**。

#### 例如一个类

```py
class temp_1():
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def print_val(self):
        print(self.a, self.b)

class temp_2():
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def print_val(self):
        print(self.a, self.b)

obj1 = temp_1(30, 40)
obj2 = temp_2(30, 40)
obj1.print_val()
obj2.print_val()

print((obj1 is obj2))

```

**输出:**

```py
30 40
30 40
False

```

**说明:**

这两个类具有相同的属性、相同的功能以及相同的参数。值得注意的区别是，它们通过内存引用是不同的。所以，通过这个代码实验得出的结论是:对于更有效的代码可测试性来说，关键字比 **==** 操作符更有用。

## 摘要

double 等于只检查值，但是，**检查**的值和引用。因此，本文向我们讲述了 Python 中“is”和“==”之间的区别，以及我们如何在 Python 编程中有效地使用这两个概念，并使我们的代码更加健壮。

### 参考

*   [https://docs.python.org/3/library/keyword.html](https://docs.python.org/3/library/keyword.html
    )
*   [https://stack overflow . com/questions/2987958/how-the-is-keyword-implemented-in-python](https://stackoverflow.com/questions/2987958/how-is-the-is-keyword-implemented-in-python)