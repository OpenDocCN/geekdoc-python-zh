# Python 按位运算符

> 原文：<https://www.askpython.com/python/python-bitwise-operators>

运算符用于对值和变量执行运算。这些符号执行各种计算。运算符所运算的值称为操作数。

在 Python 中，逐位运算符用于根据位对[整数](https://www.askpython.com/python/python-numbers)进行计算。整数被转换成`binary`，然后进行逐位运算。然后，结果以`decimal`格式存储回来。

* * *

## Python 中按位运算符的类型

|  | 操作员 | 句法 |
| --- | --- | --- |
|  | 按位与(&) | x & y |
|  | 按位或(&#124;) | x &#124; y |
|  | 按位非(~) | ~x |
|  | 按位异或(^) | x ^ y |
|  | 按位右移(>>) | x > > |
|  | 按位左移(< | x < < |

* * *

### 1.按位 AND 运算符

当两位都为 1 时，该语句返回 1，否则返回 0。

x = 5 = 0101(二进制)

y = 4 = 0100(二进制)

x & y = 0101 & 0100 = 0100 = 4(小数)

* * *

### 2.按位 OR 运算符

当两位中的任何一位为 1 时，语句返回 1，否则返回 0。

x = 5 = 0101

y = 4 = 0100

x & y = 0101 | 0100 = 0101 = 5(小数)

* * *

### 3.按位非运算符

该语句返回所提到的数的补数。

x = 5 = 0101

~x = ~0101

= -(0101 + 1)

= -(0110) = -6(十进制)

* * *

### 4.按位异或运算符

如果任一位为 1，另一位为 0，则该语句返回 true，否则返回 false。

x = 5 = 0101(二进制)

y = 4 = 0100(二进制)

x & y = 0101 ^ 0100

= 0001

= 1(小数)

* * *

## 按位移位运算符

移位运算符用于左移或右移一个数的位，从而分别将该数乘以或除以 2。当我们必须将一个数乘以或除以 2 时，就会用到它们。

### 5.按位右移运算符

它将数字的位向右移动，结果在右边的空白处填充 0。它提供了一种类似于用 2 的幂来除这个数的效果。

x = 7

x >> 1

= 3

* * *

### 6.按位左移运算符

它将数字的位向左移动，结果在左边的空白处填充 0。它提供了一种类似于将数字乘以 2 的幂的效果。

x = 7

x << 1

= 14

* * *

## Python 按位运算符示例

```py
a = 5
b = 6

# Print bitwise AND operation    
print("a & b =", a & b)  

# Print bitwise OR operation  
print("a | b =", a | b)  

# Print bitwise NOT operation   
print("~a =", ~a)  

# print bitwise XOR operation   
print("a ^ b =", a ^ b)  

c = 10
d = -10

# print bitwise right shift operator 
print("c >> 1 =", c >> 1) 
print("d >> 1 =", d >> 1) 

c = 5
d = -10

# print bitwise left shift operator 
print("c << 1 =", c << 1) 
print("d << 1 =", d << 1) 

```

**输出**:

```py
a & b = 4
a | b = 7
~a = -6
a ^ b = 3
c >> 1 = 5
d >> 1 = -5
c << 1 = 10
d << 1 = -20
```

* * *

## 参考

*   Python 按位运算符
*   [Python 组织文档](https://wiki.python.org/moin/BitwiseOperators)