# 代码片段:使用 Python 求解二次方程

> 原文：<https://www.pythoncentral.io/code-snippets-using-python-to-solve-the-quadratic-equation/>

Python 是一种通用且强大的编码语言，可用于执行各种功能和流程。感受 Python 如何工作的最好方法之一是用它来创建算法和求解方程。在这个例子中，我们将向您展示如何使用 Python 来求解一个更广为人知的数学方程:二次方程(ax ² + bx + c = 0)。

```py
import cmath

print('Solve the quadratic equation: ax**2 + bx + c = 0')
a = float(input('Please enter a : '))
b = float(input('Please enter b : '))
c = float(input('Please enter c : '))
delta = (b**2) - (4*a*c)
solution1 = (-b-cmath.sqrt(delta))/(2*a)
solution2 = (-b+cmath.sqrt(delta))/(2*a)

print('The solutions are {0} and {1}'.format(solution1,solution2))
```

正如您所看到的，为了求解方程，必须导入 cmath 模块，并且通过使用乘法、除法和 cmath.sqrt 方法(可用于求一个数的平方根)来求解方程。打印出来的文字可以定制成你喜欢说的任何话。