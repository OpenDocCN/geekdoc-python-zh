# Python 逻辑运算符中的“与”

> 原文：<https://www.askpython.com/python/and-in-python-logical-operator>

Python 有三个逻辑运算符。Python 中的逻辑运算符“and”与两个布尔操作数一起使用，并返回一个布尔值。它也被称为短路运算符或布尔运算符。我们不能在 Python 中重载“and”运算符。它只适用于布尔操作数。

## 逻辑运算符——在 Python 中

假设我们有两个布尔变量——x 和 y。只有四种可能的变化和两种可能的结果。

| **x** | **y** | **x 和 y** |
| 真实的 | 真实的 | 真实的 |
| 真实的 | 错误的 | 错误的 |
| 错误的 | 真实的 | 错误的 |
| 错误的 | 错误的 | 错误的 |

基于上表，and 运算的结果是:**如果 x 为假，则 x，否则 y** 。

让我们看一些 Python 代码中“and”操作符的简单例子。

```py
>>> x = True
>>> y = False
>>> x and y
False
>>> y = True
>>> x and y
True

```

## 按位&(与)运算符

Python 中的按位 and 运算符仅适用于整数。操作数被转换成二进制，然后对每一位执行“与”运算。然后，该值被转换回十进制并返回。

如果两位都是 1，那么& operator 返回 1，否则返回 0。让我们看一些例子。

```py
>>> 10 & 5
0
>>> 10 & -5
10

```

**解释:** 
10 = 1010
5 = 0101
-5 = 1011

1010&0101 = 0000 = 0
1010&1011 = 1010 = 10

## 摘要

Python 中的布尔运算符“and”处理布尔操作数。我们不能重载它或者使用非布尔值。我们还有按位 and 运算符，它只处理整数。

## 下一步是什么？

*   [Python 中的运算符](https://www.askpython.com/python/python-operators)
*   [Python // Operator](https://www.askpython.com/python/python-floor-division-double-slash-operator)
*   [Python 中的数字](https://www.askpython.com/python/python-numbers)
*   [Python 元组](https://www.askpython.com/python/tuple/python-tuple)
*   [Python 字典](https://www.askpython.com/python/dictionary/python-dictionary-dict-tutorial)

## 资源

*   [Python.org 文件](https://docs.python.org/3/library/stdtypes.html#boolean-operations-and-or-not)