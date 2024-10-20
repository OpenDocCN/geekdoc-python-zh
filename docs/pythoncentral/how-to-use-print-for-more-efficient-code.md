# 如何使用 Print 获得更高效的代码

> 原文：<https://www.pythoncentral.io/how-to-use-print-for-more-efficient-code/>

如果您还没有充分利用 Python 代码，这里有一个快速提示可以帮助您提高 Python 代码的效率。如果你想打印一个列表中用逗号分隔的所有值，有几种不同的方法:有复杂的方法，也有简单的方法。这里有一个复杂方式的例子:

```py
food = ["pizza", "tacos", "ice cream", "cupcakes", "burgers"]
print(', '.join(str(x) for x in food))
```

上面代码的输出将是:

```py
pizza, tacos, ice cream, cupcakes, burgers
```

print 语句中的代码并不简洁高效。如果您想以简单有效的方式打印同一列表中的所有值，请尝试使用 print 语句:

```py
print(*food, sep=", ")
```

上面的 print 语句的输出将与第一个示例中的输出完全相同:

```py
pizza, tacos, ice cream, cupcakes, burgers
```

以这种方式打印列表将确保您的代码看起来干净并且尽可能高效地执行。