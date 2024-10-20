# 将公里转换为英里的 Python 程序

> 原文：<https://www.askpython.com/python/examples/convert-kilometers-to-miles>

在本文中，我们将重点关注在 Python 中将公里转换成英里的**逐步方法。**

* * *

## 理解公里到英里转换背后的逻辑

让我们从基础开始，即理解这些测量单位的含义。公里和英里代表长度单位。

1 公里等于 0.62137 英里。

转换逻辑:

```py
Miles = kilometers * 0.62137 
OR
Kilometers = Miles / 0.62137   

```

因此，值 0.62137 可以被认为是要进行转换的转换因子或比率。

理解了转换背后的逻辑之后，现在让我们使用 Python 作为代码库来理解和实现相同的内容。

* * *

## 将公里转换成英里的简单步骤

通过下面的步骤，你会对公里和英里的换算有一个清晰的认识。

步骤 1:定义一个变量来存储公里值或接受用户的输入。

```py
kilo_meter = float(input("Enter the speed in Kilometer as a unit:\n"))

```

**第二步:定义转换系数/比值，并将其存储到变量中。**

```py
conversion_ratio = 0.621371

```

第三步:定义一个变量来存储公里转换成英里的值。进一步写出公里到英里转换的逻辑。

```py
miles = kilo_meter * conversion_ratio

```

**第四步:使用 print()函数显示转换后的值。**

```py
print("The speed value in Miles:\n", miles)

```

**完整代码:**

```py
kilo_meter = float(input("Enter the speed in Kilometer as a unit:\n"))

conversion_ratio = 0.621371

miles = kilo_meter * conversion_ratio

print("The speed value in Miles:\n", miles)

```

**输出:**

```py
Enter the speed in Kilometer as a unit:
100
The speed value in Miles:
62.137100000000004

```

另一个简单的静态定义方法可以是定义一个 [Python 函数](https://www.askpython.com/python/python-functions)来将公里转换成英里。

```py
def km_to_mile(km):
    con_ratio= 0.621371
    mile = km*con_ratio
    print("The speed value in Miles:\n", mile)

km_to_mile(100)

```

**输出:**

```py
The speed value in Miles:
 62.137100000000004

```

* * *

## 结论

到此，我们就结束了这个话题。如果你遇到任何疑问，请随时在下面发表评论，是的，继续尝试这样的编程问题，以增强和提高你的技能。

更多关于 Python 编程的帖子，请访问 [Python @ AskPython](https://www.askpython.com/) 。

* * *

## 参考

*   [Python 编程—文档](https://docs.python.org/3/tutorial/)