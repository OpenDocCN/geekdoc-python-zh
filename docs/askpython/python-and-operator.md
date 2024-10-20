# Python 和运算符

> 原文：<https://www.askpython.com/python/python-and-operator>

运算符基本上用于对要操作的数据执行操作。有各种运算符，即逻辑运算符、按位运算符、算术运算符等。

Python 中有两种 AND 运算符:

1.  `Logical AND Operator`
2.  `Bitwise AND Operator`

* * *

## 逻辑 AND 运算符

逻辑 AND 运算符处理布尔值，并根据条件得出 True 或 False。当两个操作数都为真时，**和**运算符返回真，否则返回假。

**语法:**

操作 1 和操作 2

**举例**:

```py
num1 = int(input('Enter first number:\n'))
num2 = int(input('Enter second number:\n'))
if num1 > num2 and num1 < 10:
    print('True')
elif num1 < 0 and num2 < 0:
    print('Both the numbers are negative.')
else:
    print('False')

```

**输出**:

输入第一个数字:
9
输入第二个数字:
5
真

* * *

## 逻辑运算符重载

Python 逻辑运算符处理布尔值。默认情况下，对象布尔值为**真**。如果对象为**无**或**假**，那么布尔值为**假**。我们可以提供 **__bool__()** 实现来改变一个对象的默认布尔值。

```py
class Data:

    def __init__(self, i):
        self.id = i

    def __bool__(self):
        return self.id % 2 == 0

d1 = Data(6)
d2 = Data(4)

print(bool(Data(3)) and bool(Data(4)))  # False

```

上面的代码片段将打印 False，因为 Data(3)布尔值为 False。

如果我们删除 __bool__()函数实现，两个数据对象的布尔值都将为真，并且它将打印为真。

* * *

## 按位 AND 运算符

按位运算符对位进行运算。当操作数的两位都为 1 时，它返回 1，否则它返回 0(零)。

**语法:**

操作数 1 和操作数 2

a = 6 = 0110(二进制)

b = 4 = 0100(二进制)

a & b = 0110 & 0100 = 0100 = 4(十进制)

**举例**:

```py
a = 6
b = 4

c = a & b

print(c)

```

**输出**:

four

* * *

## 按位运算符重载

Python 中有各种定义的方法来重载按位运算符。

| 按位运算符 | 句法 |
| & | __ 和 _ _(自己，其他) |
| &#124; | __ 或 _ _(自己，其他) |
| ^ | __xor__(自身，其他) |
| ~ | __invert__(self) |
| << | __lshift__(自己，其他) |
| >> | __rshift__(自己，其他) |

```py
class Operate(): 
    def __init__(self, x): 
        self.x = x 

    def __and__(self, res): 
        print("Bitwise AND operator overloading Example") 
        if isinstance(res, Operate): 
            return self.x & res.x 
        else: 
            raise ValueError("Must belong to Operate Class") 

if __name__ == "__main__": 
    a = Operate(6) 
    b = Operate(4) 
    print(a&b) 

```

**输出**:

按位 AND 运算符重载示例
4

* * *

## 逻辑运算符的求值顺序

通过操作符执行操作数从`left to right`开始。

**举例**:

```py
def evaluate(n): 
    print("Function called for operand:", n) 
    return True if n > 0 else False

x = evaluate 
y = evaluate 
z = evaluate

if x(-1) or y(5) and z(10): 
    print("One of the numbers is positive.") 

```

**输出**:

为操作数调用的函数:5
为操作数调用的函数:10
其中一个数字是正数。

* * *

## 参考

*   Python 按位运算符
*   Python 逻辑运算符
*   [Python 操作者文档](https://docs.python.org/3/library/operator.html)