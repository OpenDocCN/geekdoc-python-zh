# 用 Python 计算 a^n:用 Python 计算幂的不同方法

> 原文：<https://www.askpython.com/python/examples/compute-raised-to-power>

在本教程中，我们将以几种不同的方式计算 **`a`的`n`** 次方。让我们一步一步地看一个又一个方法。

***也读作: [Python pow()方法](https://www.askpython.com/python/built-in-methods/python-pow)***

* * *

## 方法 1:基本方法

计算`a^n`最基本的方法就是将**乘以数字 a，n 乘以**反复计算。这种方法非常慢，效率也不高。

尽管如此，这种方法的代码还是在下面提到了。

```py
def basic_approach(a,n):
    ans = 1
    for i in range(n):
        ans *= a
    return ans

print(basic_approach(2,5))

```

我们从上面提到的代码得到的输出是 **32** ，这是正确的输出。现在让我们转到下一个方法。

* * *

## 方法 2:普通递归方法

我们将通过递归来处理这个方法。如果你想知道更多关于递归的知识，你可以阅读下面提到的教程。

***了解更多关于递归的知识:[Python 中的递归](https://www.askpython.com/python/python-recursion-function)***

这里的基本概念是 **fun(a，n) = a * fun(a，n-1)** 。所以递归可以用来计算 a 的 n 次幂。

代码如下所述。添加评论供您参考。

```py
def normal_recursion(a,n):

    # If power is 0 : a^0 = 1
    if(n==0):
        return 1

    # If power is 1 : a^1 = a
    elif(n==1):
        return a

    # For n>=2 : a^n = a* (a^(n-1))
    term = normal_recursion(a,n-1)
    term = a * term

    # Return the answer
    return term

print(normal_recursion(2,5))

```

我们从上面的代码中得到的输出是 **32** ，这是准确无误的输出。让我们转到下一个方法，它只使用递归，但以更好的方式。

* * *

## 方法 3:快速递归方法

前面我们使用了线性递归方法，但是计算 n 的幂也可以基于 n 的值(幂值)来计算。

1.  如果 **n 是偶数**那么 **fun(a，n)=【fun(a，n/2)】^ 2**
2.  如果 **n 是奇数**那么 **fun(a，n)= a *(fun(a，n/2)】^ 2)**

这将是一种更有效的方法，并将在很大程度上减少程序所花费的时间。下面提到了相同方法的代码。

```py
def fast_recursion(a,n):

    # If power is 0 : a^0 = 1
    if(n==0):
        return 1

    # If power is 1 : a^1 = a
    elif(n==1):
        return a

    # For n>=2 : n can be even or odd

    # If n is even : a^n = (a^(n/2))^2
    # if n is odd : a^n = a * ((a^(n/2))^2)

    # In both the cases we have the calculate the n/2 term
    term = fast_recursion(a,int(n/2))
    term *= term

    # Now lets check if n is even or odd
    if(n%2==0):
        return term
    else:
        return a*term

print(fast_recursion(2,5))

```

该代码的输出也是正确的 **32** 。与以前的方法相比，这种方法占用一半的时间。

* * *

## 结论

因此，在本教程中，我们学习了如何使用各种方法计算 a 的 n 次幂，有些方法涉及递归，有些不涉及。你可以采用任何一种方法，但选择最有效的方法总是更好。

感谢您的阅读！编码快乐！👩‍💻

* * *