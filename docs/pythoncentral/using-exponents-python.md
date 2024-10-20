# 在 Python 中使用指数

> 原文：<https://www.pythoncentral.io/using-exponents-python/>

到目前为止，您可能已经知道如何用 Python 对[和](https://www.pythoncentral.io/multiplying-dividing-numbers-python/)进行乘除运算。Python 中的乘法相当简单，也很容易做到。但是使用指数呢？例如，你如何计算一个数的二次幂？如果你不确定，你可能会发现答案很简单。要计算一个数的另一个数的幂，需要使用“**”运算符。两个数相乘只使用一个*符号，一个数的另一个数的幂的运算符使用两个:*。

让我们看一个例子。要找到 4 的平方(4 的 2 次方是另一种说法)，您的代码应该是这样的:

```py
4**2
```

很简单，对吧？

要打印上面等式的结果，别忘了使用 print 命令。：

```py
print(4**2)
```

代码的输出将是:

```py
16
```

下面的片段将会给你一个我们如何在真实环境中使用指数的例子。在代码片段中，我们使用匿名函数(lambda)对数字 0-5 进行二次幂运算，并打印结果。

```py
squares = 5

result = list(map(lambda x: 2 ** x, range(terms)))

for i in range(squares):
 print("2 raised to the power of",i,"is",result[i])
```

所以上面代码片段的输出应该是:

```py
2 raised to the power of 0 is 1
2 raised to the power of 1 is 2
2 raised to the power of 2 is 4
2 raised to the power of 3 is 8
2 raised to the power of 4 is 16
2 raised to the power of 5 is 32
```

要自己使用该代码片段并获得不同的结果，请更改 squares 变量的值(这将根据您选择的数字的大小给出更多或更少的结果)，或将 2 值更改为另一个数字。如果您正在寻找一种方法来理解如何在 Python 中正确处理指数，这段代码片段是探索这种技能的一个很好的选择。