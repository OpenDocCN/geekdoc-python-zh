# Python 101:递归

> 原文：<https://www.blog.pythonlibrary.org/2017/08/10/python-101-recursion/>

递归是数学和计算机科学中的一个话题。在计算机编程语言中，术语“递归”指的是调用自身的函数。另一种说法是函数定义将函数本身包含在其定义中。当我的计算机科学教授谈到递归时，我收到的第一个警告是，您可能会意外地创建一个无限循环，这将使您的应用程序挂起。发生这种情况是因为当你使用递归时，你的函数可能会无限地调用自己。因此，与任何其他潜在的无限循环一样，您需要确保您有办法打破循环。大多数递归函数的思想是将正在进行的过程分解成更小的部分，我们仍然可以用同一个函数来处理。

描述递归的最常用方法通常是通过创建阶乘函数来说明。阶乘通常是这样的: **5！**注意数字后面有感叹号。该符号表示它将被视为阶乘。这意味着 **5！= 5*4*3*2*1** 或者 120。

让我们看一个简单的例子。

```py

# factorial.py

def factorial(number):
    if number == 0:
        return 1
    else:
        return number * factorial(number-1)

if __name__ == '__main__':
    print(factorial(3))
    print(factorial(5))

```

在这段代码中，我们检查传入的数字，看它是否等于零。如果是，我们返回数字 1。否则，我们取这个数，用它乘以调用同一个函数的结果，但是这个数减一。我们可以稍微修改一下这段代码，以获得递归的次数:

```py

def factorial(number, recursed=0):
    if number == 0:
        return 1
    else:
        print('Recursed {} time(s)'.format(recursed))
        recursed += 1
        return number * factorial(number-1, recursed)

if __name__ == '__main__':
    print(factorial(3))

```

每次我们调用 factorial 函数并且这个数大于零时，我们打印出我们递归的次数。您应该看到的最后一个字符串应该是 **"Recursed 2 time(s)"** ，因为它应该只需要用数字 3 调用 factorial 两次。

* * *

### Python 的递归极限

在本文的开始，我提到你可以创建一个无限递归循环。在某些语言中可以，但是 Python 实际上有递归限制。您可以通过执行以下操作自行检查:

```py

>>> import sys
>>> sys.getrecursionlimit()
1000

```

如果您觉得这个限制对您的程序来说太低，您也可以通过 sys 模块的 **setrecursionlimit()** 函数来设置递归限制。让我们试着创建一个超过这个限制的递归函数，看看会发生什么:

```py

# bad_recursion.py

def recursive():
    recursive()

if __name__ == '__main__':
    recursive()

```

如果您运行这段代码，您应该会看到以下抛出的异常: **RuntimeError:超过了最大递归深度**

Python 阻止你创建一个以永无止境的递归循环结束的函数。

* * *

### 用递归展平列表

除了阶乘，你还可以用递归做其他事情。更实际的例子是创建一个函数来展平嵌套列表，例如:

```py

# flatten.py

def flatten(a_list, flat_list=None):
    if flat_list is None:
        flat_list = []

    for item in a_list:
        if isinstance(item, list):
            flatten(item, flat_list)
        else:
            flat_list.append(item)

    return flat_list

if __name__ == '__main__':
    nested = [1, 2, 3, [4, 5], 6]
    x = flatten(nested)
    print(x)

```

当你运行这段代码时，你应该得到一个整数列表，而不是一个整数列表和一个列表。当然，还有许多其他有效的方法来展平嵌套列表，比如使用 Python 的 **itertools.chain()** 。您可能想要检查 chain()类背后的代码，因为它有一种非常不同的方法来展平列表。

* * *

### 包扎

现在，您应该对递归的工作原理以及如何在 Python 中使用它有了基本的了解。我认为 Python 对递归有一个内置的限制，以防止开发人员创建结构不良的递归函数，这很好。我还想指出，在我多年的开发生涯中，我不认为我真的需要使用递归来解决问题。我肯定有很多问题的解决方案可以在递归函数中实现，但是 Python 有如此多的其他方法来做同样的事情，我从来没有觉得有必要这样做。我想提出的另一个注意事项是，递归可能很难调试，因为很难判断错误发生时你已经达到了什么级别的递归。

不管怎样，我希望这篇文章对你有用。编码快乐！

* * *

### 相关阅读

*   如何像计算机科学家一样思考- [第三章](http://openbookproject.net/thinkcs/python/english3e/recursion.html)
*   StackOverflow - [如何用 python 构建递归函数？](https://stackoverflow.com/q/479343/393194)
*   Python 练习册- [函数式编程](http://anandology.com/python-practice-book/functional-programming.html)