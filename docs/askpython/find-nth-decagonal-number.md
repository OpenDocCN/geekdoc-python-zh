# 如何在 Python 中求第 n 个十边形数？

> 原文：<https://www.askpython.com/python/examples/find-nth-decagonal-number>

嘿伙计们！在本教程中，我们将探索如何在 Python 中获得第 n 个十边形数。但是首先，让我们复习十次数的基础知识。

* * *

## **十次数介绍**

十边形数是一种图形数，它将三角形和正方形的概念扩展到十边形(十边形)。第 n 个十边形数计算 n 个嵌套十边形图案中的点数，每个嵌套十边形都有一个共享角。

以下公式生成第 n 个十进制数:

***d(n)= 4*n^2–3 * n***

* * *

## **求第 n 个十次数的算法**

为了使用 Python 编程语言获得第 n 个十进制数，我们将遵循下面提到的步骤:

1.  取 n 的输入值
2.  使用上一节提到的公式计算 D(n)的值。
3.  显示 D(n)的计算值

* * *

## **用 Python 寻找第 n 个十次方数**

```py
def GetDecagonalNumber(n):
    return (4*n*n - 3*n)

n = int(input("Enter the value of n: "))
print("The required Decagonal Number is: ", GetDecagonalNumber(n))

```

这里，我们创建了一个函数，使事情更容易理解，并概括了一个可以在任何其他代码中重用的直接函数。

* * *

**样本输出**

```py
Enter the value of n: 3
The required Decagonal Number is:  27

```

```py
Enter the value of n: 20
The required Decagonal Number is:  1540

```

* * *

## **结论**

恭喜你！您刚刚学习了如何在 Python 编程语言中计算第 n 个十次方数。希望你喜欢它！😇

喜欢这个教程吗？无论如何，我建议你看一下下面提到的教程:

1.  [Python 中的阿姆斯特朗数——易于实现](https://www.askpython.com/python/examples/armstrong-number)
2.  [Python 中的 Harshad 数字–易于实现](https://www.askpython.com/python/examples/harshad-number)

感谢您抽出时间！希望你学到了新的东西！！😄

* * *