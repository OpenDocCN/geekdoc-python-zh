# 在 Python 中检查质数

> 原文：<https://www.pythonforbeginners.com/basics/check-for-prime-number-in-python>

质数是那些只有两个因子的数，即 1 和数本身。在本文中，我们将讨论在 python 中检查素数的两种方法。

## 什么是质数？

质数是那些大于一且只有两个因子的正整数。质数的例子有 2、3、5、7、11、13、17、19、23、29 等。

这里，

*   2 只有两个因子，即 1 和 2。
*   3 只有两个因子，即 1 和 3。
*   5 只有两个因子，即 1 和 5。

你可以观察到所有其他数字也只有两个因素。

## 在 Python 中检查质数

为了检验一个数是否是质数，我们只需确定所有大于 1 小于这个数本身的数都不是这个数的因数。为此，我们将定义一个函数 isPrime()，它接受数字 N 作为输入。然后使用循环的[检查 2 和 N-1 之间的任何数是否是 N 的因数。如果存在一个因子，它将返回 False，表明输入的数字 N 不是质数。否则，它返回 True。](https://www.pythonforbeginners.com/control-flow-2/python-for-and-while-loops)

```py
def isPrime(N):
    for number in range(2, N):
        if N % number == 0:
            return False
    return True

input_number = 23
output = isPrime(input_number)
print("{} is a Prime number:{}".format(input_number, output))
input_number = 126
output = isPrime(input_number)
print("{} is a Prime number:{}".format(input_number, output)) 
```

输出:

```py
23 is a Prime number:True
126 is a Prime number:False
```

在上面的例子中，我们已经检查了从 2 到 N-1 的每一个数，看它是否是 N 的因子。我们可以通过检查数字直到 N/2 而不是 N-1 来优化这个过程。这是因为 N 大于 N/2 的唯一因素是 N 本身。因此，我们将只检查 N/2 之前的数字因子。此外，我们还可以检查一个数字是否是偶数。如果大于 2 的数是偶数，它就永远不是质数。我们可以使用下面给出的这些概念定义一个改进的 isPrime()函数。

```py
def isPrime(N):
    for number in range(2, N//2):
        if N % number == 0:
            return False
    return True

input_number = 23
output = isPrime(input_number)
print("{} is a Prime number:{}".format(input_number, output))
input_number = 126
output = isPrime(input_number)
print("{} is a Prime number:{}".format(input_number, output)) 
```

输出:

```py
23 is a Prime number:True
126 is a Prime number:False
```

我们可以使用简单的逻辑再次优化上面的程序。你可以观察到一个数的因子总是成对出现。对于一个数 N，因子可以配对为(1，N)，(2，N/2)，(3，N/3)，(4，N/4)直到(N1/2，N^(1/2) )。因此，为了检查因子，我们可以只检查到 N^(1/2) 而不是 N/2。

例如，如果给我们一个数字 100，所有的因子都可以配对成(1，100)、(2，50)、(4，25)、(5，20)和(10，10)。在这里，如果 100 能被 2 整除，就一定能被 50 整除，如果 100 能被 4 整除，就一定能被 25 整除。我们不需要显式地检查一对数字中的两个数字来检查一个数字是否是因子。因此，为了检查质数，我们可以简单地使用一个 [while 循环](https://www.pythonforbeginners.com/loops/python-while-loop)来检查一个因子直到 N^(1/2) 而不是 N/2。如果一个因子不在 2 和 N^(1/2)之间，这个数必须是一个质数。使用这个逻辑，我们可以如下修改上面示例中使用的 isPrime()函数。

```py
def isPrime(N):
    count = 2
    while count ** 2 <= N:
        if N % count == 0:
            return False
        count = count + 1
    return True

input_number = 23
output = isPrime(input_number)
print("{} is a Prime number:{}".format(input_number, output))
input_number = 126
output = isPrime(input_number)
print("{} is a Prime number:{}".format(input_number, output)) 
```

输出:

```py
23 is a Prime number:True
126 is a Prime number:False
```

## 结论

在本文中，我们讨论了 python 中检查素数的三种方法。要了解更多关于数字的知识，你可以阅读这篇关于 python 中的[复数的文章。你可能也会喜欢这篇关于 Python 中的十进制数](https://www.pythonforbeginners.com/data-types/complex-numbers-in-python)的文章。