# FizzBuzz 问题——用 Python 实现 FizzBuzz 算法

> 原文：<https://www.askpython.com/python/examples/fizzbuzz-algorithm>

FizzBuzz 算法是编码面试中的一个热门问题。Fizz 和 Buzz 是 3 和 5 的倍数。

在本教程中，我将向您展示如何使用 Python 编程语言来创建 FizzBuzz 算法。

* * *

## FizzBuzz 算法

FizzBuzz 算法的灵感来自一个儿童游戏。长期以来，这种方法一直是最流行的编码面试问题之一。

在这个问题中，给定一个数字范围，您必须使用以下规则创建输出:

1.  如果数字(x)能被 3 整除，那么结果一定是“嘶嘶”
2.  如果数字(x)能被 5 整除，那么结果一定是“嗡嗡”
3.  如果数字(x)能被 3 和 5 整除，那么结果一定是“嘶嘶作响”

这种编码问题在数字 3 和 5 中很常见，但是，您可能会遇到更复杂的数字，但是解决问题的原因是相同的。

* * *

## 使用 Python 的 FizzBuzz 算法

为了解决 FizzBuzz 问题，我们将遵循以下步骤:

1.  现在我们只考虑正整数，所以我们将使用一个 [while](https://www.askpython.com/course/python-course-while-loop) 循环，直到用户输入正整数。
2.  现在我们将使用一个[来代替](https://www.askpython.com/course/python-course-for-loop)从 1 到 n 的循环。
    *   每当我们遇到 3 和 5 的倍数时，我们就会输出‘fizz buzz’
    *   对于 3 的倍数，我们打印“Fizz”
    *   同样，对于 5 的倍数，我们显示单词“Buzz”

```py
n = -1
while(n<0):
    n = int(input("Enter the ending integer: "))

for i in range(1, n+1):
    if i % 3 == 0 and i % 5 == 0:
        print("FizzBuzz",end=" ")
    elif i % 3 == 0:
        print("Fizz", end= " ")
    elif i % 5 == 0:
        print("Buzz", end = " ")
    else:
        print(i, end = " ")

```

* * *

## 样本输出

```py
Enter the ending integer: 20
1 2 Fizz 4 Buzz Fizz 7 8 Fizz Buzz 11 Fizz 13 14 FizzBuzz 16 17 Fizz 19 Buzz

```

```py
Enter the ending integer: 100
1 2 Fizz 4 Buzz Fizz 7 8 Fizz Buzz 11 Fizz 13 14 FizzBuzz 16 17 Fizz 19 Buzz Fizz 22 23 Fizz Buzz 26 Fizz 28 29 FizzBuzz 31 32 Fizz 34 Buzz Fizz 37 38 Fizz Buzz 41 Fizz 43 44 FizzBuzz 46 47 Fizz 49 Buzz Fizz 52 53 Fizz Buzz 56 Fizz 58 59 FizzBuzz 61 62 Fizz 64 Buzz Fizz 67 68 Fizz Buzz 71 Fizz 73 74 FizzBuzz 76 77 Fizz 79 Buzz Fizz 82 83 Fizz Buzz 86 Fizz 88 89 FizzBuzz 91 92 Fizz 94 Buzz Fizz 97 98 Fizz Buzz 

```

* * *

## 结论

能被 3 和 5 整除的数字被称为嘶嘶声和嗡嗡声。如果一个数能被 3 整除，就用“Fizz”代替；如果能被 5 整除，就用“Buzz”代替；如果能被 3 和 5 整除，就用“FizzBuzz”代替。

我希望您喜欢这篇关于 Python 编程语言实现 FizzBuzz 算法的教程。

快乐学习！😇下面包含更多教程:

1.  [用 Python 解决梯子问题](https://www.askpython.com/python/examples/ladders-problem)
2.  [使用递归在 Python 中求解 0-1 背包问题](https://www.askpython.com/python/examples/knapsack-problem-recursion)
3.  [在 Python 中解决平铺问题](https://www.askpython.com/python/examples/tiling-problem)
4.  [用 Python 解决朋友旅行问题【谷歌面试问题】](https://www.askpython.com/python/examples/friends-travel-problem)

* * *