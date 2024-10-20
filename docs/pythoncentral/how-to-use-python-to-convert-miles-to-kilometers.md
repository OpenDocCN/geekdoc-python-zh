# 如何使用 Python 将英里转换成公里

> 原文：<https://www.pythoncentral.io/how-to-use-python-to-convert-miles-to-kilometers/>

令人沮丧的是，无论是测量距离、重量、温度等，整个地球都无法在一个测量系统上达成一致...每个人都行驶在道路的哪一边也是如此)。如果每个国家都使用相同的计量单位，那么这些公式将是无用和过时的，但在此之前，了解如何将英里转换为公里可能是一个好主意，反之亦然。如果你是一个 Python 开发者，那么知道如何用 Python 写公式可能会特别有用。继续阅读，看看它是如何做到的。

将英里换算成公里并不特别困难。为了在你的头脑中做一个粗略的估计，你真正需要记住的是一英里大约等于 1.6 公里(或者一公里大约是一英里的 2/3)。当试图使用公式找到正确的换算时，我们必须使用更精确的换算系数，它等于 0.62137119。

要把英里换算成公里，公式非常简单。你需要做的就是用里程数除以换算系数。要查看用 Python 写出来的效果，请查看下面的示例:

```py
miles = 30
conversion_factor = 0.62137119

kilometers = miles / conversion_factor
print kilometers
```

在上面的例子中，您只需要声明两个变量:miles 和 conversion_factor，miles 可以是您需要的任何数字，conversion _ factor 的值必须始终保持不变，以便获得正确和准确的转换。上面代码的输出将是 48.2803202，因此 30 英里大约等于 48 公里。

要将公里转换为英里，您将采用类似的方法。进行此转换时，您需要将公里值乘以转换系数。有关在 Python 中进行这种转换的示例，请参见下面的示例:

```py
kilometers = 6
conversion_factor = 0.62137119

miles = kilometers / conversion_factor
print miles
```

上面示例的输出将是 3.72822714，因此 6 公里大约等于 3 加 3