# Python 代码片段:求解二次方程

> 原文：<https://www.pythoncentral.io/python-code-snippets-solving-quadratic-equation/>

这里有一个使用 Python 的有趣方法:在一个函数中输入任意三个数字，你可以在毫秒内求解二次方程。我打赌你希望你在代数一课上就知道这个。

这个函数是由 [Programiz](https://www.programiz.com/python-programming/examples/quadratic-roots) 的人编写的，使用 Python 中的基本乘法和除法运算来求解二次方程。如果你不记得，要解二次方程，你必须取 b 的倒数，加上或减去 b 平方的平方根，减去 4 乘以 a 乘以 c 除以(除以)2 乘以 a。在二次方程中，a，b 和 c 可以是任何整数，正数或负数，只要 a 不等于零。在下面的例子中，a 等于 1，b 等于 5，c 等于 6，但是你可以把它们设置成你喜欢的任何数字。

以下是片段:

```py
a = 1
b = 5
c = 6

# To take coefficient input from the users
# a = float(input('Enter a: '))
# b = float(input('Enter b: '))
# c = float(input('Enter c: '))

# calculate the discriminant
d = (b**2) - (4*a*c)

# find two solutions
sol1 = (-b-cmath.sqrt(d))/(2*a)
sol2 = (-b+cmath.sqrt(d))/(2*a)
```

正如你所看到的，这个代码片段找到了两个可能的解决方案(正如它的本意)，因为你应该取 b 加上*或*减去 b 的平方的倒数，等等，这个等式解释了这一点。

该代码片段的输出如下:

```py
print('The solution are {0} and {1}'.format(sol1,sol2))
```

现在你有了公式，加上你自己的数字，看看这个公式是否适合你。