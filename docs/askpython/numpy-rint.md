# numpy . rint()–将 NumPy 数组元素舍入到最接近的整数

> 原文:# t0]https://www . aspython . com/python-modules/num py/numpy-rint

大家好！在本教程中，我们将学习 NumPy `**rint**`函数。该功能简单易懂，易于使用。让我们开始吧。

## numpy.rint()是什么？

NumPy **`rint`** 是 NumPy 库提供的另一个数学函数，它将输入数字舍入到最接近的整数**。**

**让我们看看这个函数的语法。**

## **numpy.rint()的语法**

```py
numpy.rint(input) 
```

**这里，输入可以是单个数字，也可以是数字的 NumPy 数组。需要注意的是，由 **`numpy.rint()`** 函数返回的**数组具有与输入数组相同的**类型**和**形状**。****

## **使用数字打印机**

**我们已经完成了语法部分:)。现在让我们写一些代码来更好地理解这个函数。**

### **numpy.rint()用于输入单个数字**

```py
import numpy as np

# Rounding the input values to the nearest integer
print("Rounded Value of 0.5 is:",np.rint(0.5))

print("Rounded Value of 1.5 is:",np.rint(1.5))

print("Rounded Value of 1.74 is:",np.rint(1.74))

print("Rounded Value of 5.56 is:",np.rint(5.56))

print("Rounded Value of 5.243 is:",np.rint(5.243))

print("Rounded Value of 10.111 is:",np.rint(10.111))

print("Rounded Value of 10.179 is:",np.rint(10.179)) 
```

****输出****

```py
Rounded Value of 0.5 is: 0.0
Rounded Value of 1.5 is: 2.0
Rounded Value of 1.74 is: 2.0
Rounded Value of 5.56 is: 6.0
Rounded Value of 5.243 is: 5.0
Rounded Value of 10.111 is: 10.0
Rounded Value of 10.179 is: 10.0 
```

**有一件有趣的事情需要观察，那就是 **`numpy.rint()`** 函数的输出总是一个**整数值**。现在让我们尝试使用 NumPy 数组的函数。**

### **NumPy rint 代表 NumPy 数字数组**

```py
import numpy as np

a = np.array((0.52 , 2.55 , 4.51 , 7.56 , 19.32))

print("Input Array:\n",a)
print("Rounded Values:\n",np.rint(a)) 
```

****输出****

```py
Input Array:
 [ 0.52  2.55  4.51  7.56 19.32]
Rounded Values:
 [ 1\.  3\.  5\.  8\. 19.] 
```

**输出数组非常明显，其中输入数组的元素被舍入到最接近的整数值。**

**这就是关于 NumPy rint 函数的全部内容。请点击查看关于 python 其他主题的精彩文章[。](https://www.askpython.com/)**

## **参考**

**[num py documentation–num py prints](https://numpy.org/doc/stable/reference/generated/numpy.rint.html)**