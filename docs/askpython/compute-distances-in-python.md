# Python 中如何计算距离？[简单的分步指南]

> 原文：<https://www.askpython.com/python/examples/compute-distances-in-python>

你好。今天我们将学习如何用 python 编程语言计算距离。在本教程中，我们将计算以下距离:

1.  汉娩距
2.  欧几里得距离
3.  曼哈顿距离

我们将查看每个距离计算的公式，然后学习如何在 python 代码的帮助下进行计算。

***也读:[用 Python 计算 a^n:用 Python 计算幂的不同方式](https://www.askpython.com/python/examples/compute-raised-to-power)***

* * *

## 用 Python 计算汉明距离

汉明距离是以二进制格式计算两个数字之间的距离。它基本上意味着二进制格式中两个数之间的位数不同。

例如，如果我们选择二进制数 101 和 111，那么它们之间的汉明距离是 1，因为它们只相差一个二进制数字。

### 用 Python 实现汉明距离

现在，为了计算不同的位数，我们将使用 XOR 运算。XOR 仅在比特不同时产生 1，否则产生 0。最后，我们将计算这两个数字的 XOR 运算中的置位位数。

```py
a = int(input())
b = int(input())

x = a^b
final_ans = 0;

while (x > 0):
    final_ans += x & 1;
    x >>= 1;

print("First Number: ",a)
print("Second Number: ",b)
print("Hamming Distance: ",final_ans)

```

我们输入 12 和 9 作为两个输入，汉明距离为 3，如下图所示。

```py
First Number:  9
Second Number:  14
Hamming Distance:  3

```

* * *

## 在 Python 中计算欧几里德距离

欧几里得距离是空间中两点之间的距离，可以借助毕达哥拉斯公式来测量。公式如下所示:

把这些点看作是(x，y，z)和(a，b，c)，那么距离计算如下:
[(x-a)^2+(y-b)^2+(z-c)^2)]的平方根。

### 实施

为了计算两个坐标点之间的欧几里德距离，我们将使用 python 中的 **numpy** 模块。

```py
import numpy as np
p1 = np.array((1,2,3))
p2 = np.array((3,2,1))
sq = np.sum(np.square(p1 - p2))
print(np.sqrt(sq))

```

上面提到的代码的输出结果是 2.8284271247461903。你也可以用计算器手动计算距离，结果大致相同。

***也读作:[计算未加权图中节点间的距离](https://www.askpython.com/python/examples/distance-between-nodes-unweighted-graph)***

* * *

## 用 Python 计算曼哈顿距离

两个向量/数组(比如说 *A* 和 *B)* 之间的曼哈顿距离被计算为σ| A[I]–B[I]|其中 A [i] 是第一个数组中的第 I 个元素，B [i] 是第二个数组中的第 I 个元素。

### 代码实现

```py
A = [1,2,3]
B = [5,3,2]

dis = 0

for i in range(len(A)):
    dis += abs(A[i] - B[i])

print("First Array is: ", A)
print("Second Array is: ", B)
print("Manhattan Distance is: ", dis)

```

上面提到的代码的输出如下所示。

```py
First Array is:  [1, 2, 3]
Second Array is:  [5, 3, 2]
Manhattan Distance is:  6

```

* * *

## 结论

我希望你理解了教程中提到的所有距离计算的概念和代码逻辑。感谢您阅读教程！

快乐学习！😇

* * *