# python//operator–基于楼层的部门

> 原文：<https://www.askpython.com/python/python-floor-based-division>

Python 3 中的`//`操作符用于执行基于楼层的划分。

这意味着`a // b`首先将 a 除以 b 得到整数商，同时丢弃余数。这意味着`a//b`的结果总是一个整数。

## Python //运算符示例

这里有几个例子来说明这一点:

```py
>>> 2 // 3
0
>>> 1.3 // 2
0.0
>>> 1.3 // 1.0
1.0
>>> 3.4 // 1.1
3.0
>>> 3.4 // 1.2
2.0
>>> -1//2
-1
>>> -6 // 2
-3
>>> -6 // -3
2

```

这显示了`//`操作符如何通过只考虑除法的整数部分来执行基于底数的除法，即使对于浮点数也是如此。

对不支持的类型(比如列表和字符串)执行这个操作，将会产生一个`TypeError`，对于任何其他算术运算符来说都是一样的。

* * *

## 重载//运算符

`//`默认是指`__floordiv__()`运算符，所以你可以通过重写这个方法(`operator.__floordiv__(a, b)`来执行[运算符重载](https://www.askpython.com/python/operator-overloading-in-python)

下面是一个重载具有相同长度的整数列表的`//`方法的例子，通过对每对元素执行单独的基于下限的除法。

所以两个整数列表`[3, 4, 5]`和`[2, 2, 1]`会给出`[3//2, 4//2, 5//1]`，简单来说就是列表`[1, 2, 5]`。

```py
import operator

class MyClass():
    def __init__(self, a):
        self.a = a

    def __floordiv__(self, b):
        if isinstance(self.a, list) and isinstance(b.a, list) and len(self.a) == len(b.a):
            result = []
            # Overload // operator for Integer lists
            for i, j in zip(self.a, b.a):
                result.append(i // j)
            return result
        else:
            # Perform Default // operation otherwise
            return operator.__floordiv__(self.a, b.a)

m = MyClass([3, 4, 5])
n = MyClass([2, 2, 1])

print(m // n)

```

输出

```py
[1, 2, 5]

```

* * *

## 结论

在本文中，我们学习了`//`地板除法运算符。我们还学习了通过实现`operator.__floordiv__(a, b)`来执行操作符重载。

## 参考

*   [Python 操作符文档](https://docs.python.org/3/library/operator.html)