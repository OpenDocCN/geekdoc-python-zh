# Python 中的几何级数

> 原文：<https://www.askpython.com/python/examples/geometric-progression-in-python>

嘿伙计们！在本教程中，我们将了解什么是几何级数，以及如何在 Python 编程语言中实现几何级数。

* * *

## 几何级数导论

几何级数是一系列的元素，其中下一项是通过将前一项乘以公比而获得的。

G.P .数列是一个数列，其中任何连续整数(项)的公比总是相同的。

这个 G.P .级数的和是基于一个数学公式。

```py
Sn = a(r^n)/(1-r)
Tn = ar^((n-1))
```

* * *

## Python 的几何进步

让我们来了解一下 Python 中的几何级数是如何工作的。为了更好地理解，我们来看两个不同的例子。

### **1。打印几何级数的前 n 项**

实现 n GP 条款涉及许多步骤。步骤如下:

**第一步**——取 a(第一项)、r(公比)和 n(项数)的输入
**第二步**——从 1 到 n+1 进行循环，在每次迭代中计算第 n 项，并一直打印这些项。

```py
# 1\. Take input of 'a','r' and 'n'
a = int(input("Enter the value of a: "))
r = int(input("Enter the value of r: "))
n = int(input("Enter the value of n: "))

# 2\. Loop for n terms
for i in range(1,n+1):
    t_n = a * r**(i-1)
    print(t_n)

```

```py
Enter the value of a: 1
Enter the value of r: 2
Enter the value of n: 10
1
2
4
8
16
32
64
128
256
512

```

* * *

### **2。获取几何级数中前 n 项的和**

计算前 n 个 GP 项的和需要几个步骤。步骤如下:

****第一步****——取 a(第一项)、r(公比)、n(项数)的输入
**第二步**——用上面提到的公式计算前‘n’项之和。

```py
# 1\. Take input of 'a','r' and 'n'
a = int(input("Enter the value of a: "))
r = int(input("Enter the value of r: "))
n = int(input("Enter the value of n: "))

if(r>1):
  S_n = (a*(r**n))/(r-1)
else:
  S_n = (a*(r**n))/(1-r)

print("Sum of n terms: ",S_n)

```

```py
Enter the value of a: 1
Enter the value of r: 2
Enter the value of n: 5
Sum of n terms:  32.0

```

* * *

## **结论**

恭喜你！您刚刚学习了如何在 Python 中实现几何级数。希望你喜欢它！😇

喜欢这个教程吗？无论如何，我建议你看一下下面提到的教程:

1.  [Python 中的记忆化——简介](https://www.askpython.com/python/examples/memoization-in-python)
2.  [Python 中的字谜简介](https://www.askpython.com/python/examples/anagrams-in-python)
3.  [Python Wonderwords 模块——简介](https://www.askpython.com/python-modules/wonderwords-module)

感谢您抽出时间！希望你学到了新的东西！！😄

* * *