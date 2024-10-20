# Python 101:三元运算符

> 原文：<https://www.blog.pythonlibrary.org/2012/08/29/python-101-the-ternary-operator/>

有很多计算机语言都包含三元(或第三)运算符，这基本上是 Python 中的一行条件表达式。如果你感兴趣，你可以在维基百科上阅读其他语言的各种翻译方式。在这里，我们将花一些时间来看几个不同的例子，以及为什么你可能会在现实生活中使用这个操作符。

我记得在 C++中使用三元运算符时，它是一个问号。我用谷歌查了一下，在 StackOverflow 问答和前面提到的维基百科的例子中找到了一些很好的例子。让我们来看看其中的一些，看看我们是否能解决它们。下面是一个最简单的例子:

```py
x = 5
y = 10
result = True if x > y else False

```

这基本上是这样的:结果将是**真**是 x 大于 y，否则结果是**假**。老实说，这让我想起了我见过的一些 Microsoft Excel 条件语句。有些人反对这种格式，但这就是官方的 [Python 文档](http://docs.python.org/release/3.0.1/reference/expressions.html#boolean-operations)所使用的格式。下面是如何在普通条件语句中编写它:

```py
x = 5
y = 10

if x > y:
    print("True")
else:
    print("False")

```

因此，使用三元运算符可以节省 3 行代码。无论如何，当你在一组文件上循环时，你可能想要使用这个结构，并且你想要过滤掉一些部分或行。在下一个例子中，我们将循环一些数字，检查它们是奇数还是偶数:

```py
for i in range(1, 11):
    x = i % 2
    result = "Odd" if x else "Even"
    print(f"{i} is {result}")

```

您会惊讶于检查除法语句剩余部分的频率。这是一种快速判断数字是奇数还是偶数的方法。在前面提到的 StackOverflow 链接中，有一段有趣的代码，作为示例展示给那些仍在使用 Python 2.4 或更早版本的人:

```py
# (falseValue, trueValue)[test]
>>> (0, 1)[5>6]

0
>>> (0, 1)[5<6]

1

```

这是相当丑陋的，但它做的工作。这是索引一个元组，当然是一个黑客，但这是一段有趣的代码。当然，它没有我们之前看到的新方法的短路值，所以两个值都要计算。这样做你甚至会遇到奇怪的错误，其中 True 是 False，反之亦然，所以我不推荐这样做。

用 Python 的 lambda 做三元也有几种方法。这是之前提到的维基百科条目中的一条:

```py
def true():
    print("true")
    return "truly"
def false():
    print("false")
    return "falsely"

func = lambda b,a1,a2: a1 if b else a2
func(True, true, false)()

func(False, true, false)()

```

这是一些古怪的代码，尤其是如果你不理解 lambdas 是如何工作的。基本上，lambda 是一个匿名函数。这里我们创建了两个正常的函数和一个 lambda 函数。然后我们用一个布尔值 True 和一个 False 来调用它。在第一次调用中，如下所示:如果布尔值为真，则调用 true 函数，否则调用 false 函数。第二个稍微有点混乱，因为它似乎是说如果布尔为 false，你应该调用 true 方法，但它实际上是说只有当 b 为布尔 False 时，它才会调用 False 方法。是啊，我也觉得有点困惑。

### 包扎

在下面的“补充阅读”一节中，还有几个其他的三元运算符的例子，您可以查看一下，但是现在您应该已经很好地掌握了如何使用它以及何时使用它。当我知道我需要创建一个简单的真/假条件，并且我想节省几行代码时，我个人会使用这种方法。然而，我经常倾向于使用显式而不是隐式，因为我知道我必须回来维护这些代码，而且我不喜欢总是不得不弄清楚这种奇怪的东西，所以我可能会直接写下这 4 行代码。当然，选择权在你。

### 附加阅读

*   表达式中的“条件句”( [Python 食谱](http://code.activestate.com/recipes/52310/))
*   [Python Lambda](https://www.blog.pythonlibrary.org/2010/07/19/the-python-lambda/)
*   Lambda 代替 if - [StackOverflow](http://stackoverflow.com/questions/7076703/lambda-instead-of-if-statement)
*   [愚蠢的λ把戏](http://p-nand-q.com/python/stupid_lambda_tricks.html)包括一个三元例子