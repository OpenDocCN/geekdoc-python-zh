# LBYL vs EAFP:防止或处理 Python 中的错误

> 原文：<https://realpython.com/python-lbyl-vs-eafp/>

处理错误和异常情况是编程中的常见要求。你可以在错误发生前*阻止错误*，或者在错误发生后*处理错误*。一般来说，你会有两种与这些策略相匹配的编码风格:**三思而后行** (LBYL)，和**请求原谅比请求许可容易** (EAFP)。在本教程中，您将深入探讨 Python 中 LBYL vs EAFP 的相关问题和注意事项。

通过学习 Python 的 LBYL 和 EAFP 编码风格，您将能够决定在处理代码中的错误时使用哪种策略和编码风格。

**在本教程中，您将学习如何:**

*   在你的 Python 代码中使用 **LBYL** 和 **EAFP** 风格
*   了解 LBYL vs EAFP 的**利弊**和**利弊**
*   决定何时使用 LBYL 或 EAFP

为了充分利用本教程，您应该熟悉[条件语句](https://realpython.com/python-conditional-statements/)和 [`try` … `except`](https://realpython.com/python-exceptions/#the-try-and-except-block-handling-exceptions) 语句是如何工作的。这两条语句是在 Python 中实现 LBYL 和 EAFP 编码风格的构建块。

**免费奖励:** [掌握 Python 的 5 个想法](https://realpython.com/bonus/python-mastery-course/)，这是一个面向 Python 开发者的免费课程，向您展示将 Python 技能提升到下一个水平所需的路线图和心态。

## 错误和异常情况:预防还是处理？

处理错误和异常情况是计算机编程的基本部分。错误和异常无处不在，如果您想要健壮和可靠的代码，您需要学习如何管理它们。

在处理错误和异常时，您至少可以遵循两种通用策略:

1.  **防止**错误或异常情况发生
2.  **处理发生后的**错误或异常情况

从历史上看，在错误发生之前防止错误一直是编程中最常见的策略或方法。这种方法通常依赖于[条件语句](https://realpython.com/python-conditional-statements/)，在许多编程语言中也被称为`if`语句。

当编程语言开始提供异常处理机制，如 [Java](https://realpython.com/java-vs-python/) 和 [C++](https://realpython.com/python-vs-cpp/) 中的`try` … `catch`语句，以及 Python 中的 [`try` … `except`](https://realpython.com/python-exceptions/#the-try-and-except-block-handling-exceptions) 语句时，在错误和异常发生后进行处理就出现了。然而，在 Java 和 C++中，处理异常可能是一个代价很高的操作，所以这些语言倾向于防止错误，而不是处理错误。

**注:**一[优化](https://docs.python.org/3.11/whatsnew/3.11.html#optimizations)即将到来 [Python 3.11](https://realpython.com/python311-exception-groups/) 是[零成本异常](https://github.com/python/cpython/issues/84403)。这意味着当没有异常出现时，`try`语句的开销将几乎被消除。

其他编程语言如 [C](https://realpython.com/c-for-python-programmers/) 和 [Go](https://go.dev/) 甚至没有异常处理机制。例如，Go 程序员习惯于使用条件语句来防止错误，如下例所示:

```py
func  SomeFunc(arg  int)  error  { result,  err  :=  DoSomething(arg) if  err  !=  nil  {   // Handle the error here... log.Print(err) return  err } return  nil }
```

这个假设的 Go 函数调用`DoSomething()`并将它的[返回值](https://realpython.com/python-return-statement/)存储在`result`和`err`中。`err`变量将保存函数执行过程中出现的任何错误。如果没有错误发生，那么`err`将包含`nil`，这是 Go 中的空值。

然后，`if`语句检查错误是否不同于`nil`，在这种情况下，函数继续处理错误。这种模式很常见，你会在大多数围棋程序中反复看到。

当没有出现异常时，Python 的异常处理机制非常有效。因此，在 Python 中，使用该语言的异常处理语法来处理错误和异常情况是很常见的，有时也是被鼓励的。这种做法经常让来自其他编程语言的人感到惊讶。

这对您来说意味着 Python 足够灵活和高效，您可以选择正确的策略来处理代码中的错误和异常情况。你既可以用条件语句*防止*错误，也可以用*语句`try` … `except`处理*错误。

Pythonistas 通常使用以下术语来确定这两种处理错误和异常情况的策略:

| 战略 | 术语 |
| --- | --- |
| 防止错误发生 | 三思而后行( [LBYL](https://docs.python.org/3/glossary.html#term-LBYL) ) |
| 错误发生后的处理 | 请求原谅比请求允许容易( [EAFP](https://docs.python.org/3/glossary.html#term-EAFP) ) |

在接下来的章节中，您将了解这两种策略，在 Python 和其他编程语言中也被称为**编码风格**。

具有昂贵的异常处理机制的编程语言倾向于依赖于在错误发生之前检查可能的错误。这些语言通常喜欢 LBYL 风格。相比之下，Python 在处理错误和异常情况时更有可能依赖其异常处理机制。

简要介绍了处理错误和异常的策略后，您就可以更深入地研究 Python 的 LBYL 和 EAFP 编码风格，并探索如何在代码中使用它们。

[*Remove ads*](/account/join/)

## “三思而后行”(LBYL)风格

LBYL，或三思而后行，是指你首先检查某件事是否会成功，然后只有在你知道它会成功的情况下才继续进行。Python 文档将这种编码风格定义为:

> 三思而后行。这种编码风格在进行调用或查找之前显式测试前置条件。这种风格与 EAFP 的方法形成了鲜明的对比，其特点是出现了许多`if`语句。([来源](https://docs.python.org/3/glossary.html#term-LBYL))

为了理解 LBYL 的本质，您将使用一个经典的例子来处理字典中丢失的键。

假设您有一个包含一些数据的字典，并且您想要逐个键地处理该字典。你预先知道字典可能包括一些特定的关键字。您还知道有些键可能不存在。你如何处理丢失的键而不被破解你的代码呢？

你会有几种方法来解决这个问题。首先，考虑如何使用条件语句来解决这个问题:

```py
if "possible_key" in data_dict:
    value = data_dict["possible_key"]
else:
    # Handle missing keys here...
```

在这个例子中，首先检查目标字典`data_dict`中是否存在`"possible_key"`。如果是这种情况，那么您访问密钥并将其内容分配给`value`。这样，您就防止了一个`KeyError`异常，并且您的代码不会中断。如果`"possible_key"`不在场，那么你在`else`条款中处理这个问题。

这种解决问题的方法被称为 LBYL，因为它依赖于在执行期望的动作之前检查先决条件。LBYL 是一种传统的编程风格，在这种风格下，你要确保一段代码在运行之前能够正常工作。如果你坚持这种风格，那么你将会在你的代码中有很多`if`语句。

这种做法并不是解决 Python 中缺少键问题的唯一或最常见的方法。您还可以使用 EAFP 编码风格，接下来您将了解这一点。

## “请求原谅比请求允许容易”(EAFP)风格

Grace Murray Hopper ，一位对计算机编程做出杰出贡献的美国计算机科学家先驱，提供了一条宝贵的建议和智慧，她说:

> 请求原谅比获得许可更容易。([来源](https://en.wikiquote.org/wiki/Grace_Hopper))

EAFP，或者说请求原谅比请求允许更容易，是这个建议应用于编程的具体表达。它建议你马上去做你期望的工作。如果它不起作用并且发生了异常，那么只需捕捉异常并适当地处理它。

根据 Python 的官方术语表, EAFP 编码风格有如下定义:

> 请求原谅比请求允许容易。这种常见的 Python 编码风格假设存在有效的键或属性，并在假设证明为假时捕捉异常。这种干净快速的风格的特点是存在许多`try`和`except`语句。这种技术与许多其他语言中常见的 LBYL 风格形成对比，比如 C. ( [来源](https://docs.python.org/3/glossary.html#term-EAFP))

在 Python 中，EAFP 编码风格非常流行和普遍。它有时比 LBYL 风格更受推荐。

这种受欢迎程度至少有两个激励因素:

1.  Python 中的异常处理快速而高效。
2.  对潜在问题的必要检查通常是语言本身的一部分。

正如官方定义所说，EAFP 编码风格的特点是使用`try` … `except`语句来捕捉和处理代码执行过程中可能出现的错误和异常情况。

下面是如何使用 EAFP 风格重写上一节中关于处理丢失键的示例:

```py
try:
     value = data_dict["possible_key"]
except KeyError:
    # Handle missing keys here...
```

在这个变体中，在使用它之前，您不检查密钥是否存在。相反，您可以继续尝试访问所需的密钥。如果由于某种原因，这个键不存在，那么您只需捕获`except`子句中的`KeyError`并适当地处理它。

这种风格与 LBYL 风格形成对比。它不是一直检查先决条件，而是立即运行所需的操作，并期望操作成功。

[*Remove ads*](/account/join/)

## 蟒蛇之路:LBYL 还是 EAFP？

Python 更适合 EAFP 还是 LBYL？这几种风格哪一种更有 Pythonic 风格？嗯，看起来 Python 开发者一般倾向于 EAFP 而不是 LBYL。这种行为基于几个原因，稍后您将对此进行探讨。

然而，事实仍然是 Python 作为一种语言，对于这两种编码风格没有明确的偏好。[Python 的创始人吉多·范·罗苏姆](https://twitter.com/gvanrossum)也说过:

> [……]我不同意 EAFP 比 LBYL 好，或者 Python“普遍推荐”的立场。([来源](https://mail.python.org/pipermail/python-dev/2014-March/133118.html))

正如生活中的许多其他事情一样，最初问题的答案是:视情况而定！如果眼前的问题表明 EAFP 是最好的方法，那就去做吧。另一方面，如果最佳解决方案意味着使用 LBYL，那么使用它时不要认为违反了 Pythonic 规则。

换句话说，你应该对在你的代码中使用 LBYL 或者 EAFP 持开放态度。根据您的具体问题，这两种风格都可能是正确的解决方案。

可以帮助你决定使用哪种风格的是回答这个问题:在这种情况下，什么更方便，*防止错误发生*还是*在错误发生后处理它们*？想好答案，做出选择。在接下来的部分，你将探索 LBYL 和 EAFP 的利弊，这可以帮助你做出这个决定。

## Python 中的 LBYL 和 EAFP 编码风格

为了更深入地探究何时使用 Python 的 LBYL 或 EAFP 编码风格，您将使用一些相关的比较标准来比较这两种风格:

*   支票数量
*   可读性和清晰性
*   竞争条件风险
*   代码性能

在下面的小节中，您将使用上面的标准来发现 LBYL 和 EAFP 编码风格如何影响您的代码，以及哪种风格适合您的特定用例。

### 避免不必要的重复检查

EAFP 相对于 LBYL 的优势之一是，前者通常可以帮助您避免不必要的重复检查。例如，假设您需要一个[函数](https://realpython.com/defining-your-own-python-function/)，它将正数[数字](https://realpython.com/python-numbers/)作为[字符串](https://realpython.com/python-strings/)，并将它们转换为整数值。您可以使用 LBYL 编写这个函数，如下例所示:

>>>

```py
>>> def to_integer(value):
...     if value.isdigit():
...         return int(value)
...     return None
...

>>> to_integer("42")
42

>>> to_integer("one") is None
True
```

在这个函数中，首先检查`value`是否包含可以转换成数字的内容。为了进行检查，您使用内置的 [str](https://docs.python.org/3/library/stdtypes.html#str) 类中的 [`.isdigit()`](https://docs.python.org/3/library/stdtypes.html#str.isdigit) 方法。如果输入字符串中的所有字符都是数字，这个方法返回`True`。否则返回`False`。酷！这个功能听起来是正确的选择。

如果您尝试了该功能，那么您会得出结论，它按照您的计划工作。如果输入包含数字，则返回一个整数；如果输入包含至少一个非数字字符，则返回 [`None`](https://realpython.com/null-in-python/) 。然而，在这个函数中有一些隐藏的重复。你能发现它吗？对 [`int()`](https://docs.python.org/3/library/functions.html#int) 的调用在内部执行所有需要的检查，将输入字符串转换为实际的整数。

因为检查已经是`int()`的一部分，用`.isdigit()`测试输入字符串会重复已经存在的检查。为了避免这种不必要的重复和相应的开销，您可以使用 EAFP 风格，做如下事情:

>>>

```py
>>> def to_integer(value):
...     try:
...         return int(value)
...     except ValueError:
...         return None
...

>>> to_integer("42")
42

>>> to_integer("one") is None
True
```

这个实现完全消除了您之前看到的隐藏重复。它还有其他优点，您将在本教程的后面部分探索，比如提高可读性和性能。

### 提高可读性和清晰度

要发现使用 LBYL 或 EAFP 如何影响代码的可读性和清晰性，假设您需要一个将两个数相除的函数。该函数必须能够检测其第二个参数**分母**是否等于`0`，以避免`ZeroDivisionError`异常。如果分母是`0`，那么函数将返回一个默认值，该值可以在调用中作为可选参数[提供。](https://realpython.com/python-optional-arguments/)

下面是使用 LBYL 编码风格的该函数的实现:

>>>

```py
>>> def divide(a, b, default=None):
...     if b == 0:  # Exceptional situation
...         print("zero division detected")  # Error handling
...         return default
...     return a / b  # Most common situation
...

>>> divide(8, 2)
4.0

>>> divide(8, 0)
zero division detected

>>> divide(8, 0, default=0)
zero division detected
0
```

`divide()`函数使用一个`if`语句来检查除法中的分母是否等于`0`。如果是这种情况，那么函数[将](https://realpython.com/python-print/)一条消息打印到屏幕上，并返回存储在`default`中的值，该值最初被设置为`None`。否则，该函数将两个数相除并返回结果。

上面的`divide()`实现的问题是，它将异常情况放在了前面和中心，影响了代码的可读性，并使函数不清晰和难以理解。

最后，这个函数是关于计算两个数的除法，而不是确保分母不是`0`。因此，在这种情况下，LBYL 风格会分散开发人员的注意力，将他们的注意力吸引到异常情况上，而不是主流情况上。

现在考虑如果你使用 EAFP 编码风格编写这个函数会是什么样子:

>>>

```py
>>> def divide(a, b, default=None):
...     try:
...         return a / b  # Most common situation
...     except ZeroDivisionError:  # Exceptional situation
...         print("zero division detected")  # Error handling
...         return default
...

>>> divide(8, 2)
4.0

>>> divide(8, 0)
zero division detected

>>> divide(8, 0, default=0)
zero division detected
0
```

在这个新的`divide()`实现中，函数的主要计算在`try`子句中处于前端和中心，而异常情况在后台的`except`子句中被捕获和处理。

当您开始阅读这个实现时，您会立即注意到这个函数是关于计算两个数的除法的。您还会意识到，在异常情况下，第二个参数可以等于`0`，生成一个`ZeroDivisionError`异常，这个异常在`except`代码块中得到很好的处理。

[*Remove ads*](/account/join/)

### 避免竞态条件

当不同的程序、进程或线程同时访问给定的计算资源时，就会出现**竞争条件**。在这种情况下，程序、进程或线程竞相访问所需的资源。

出现竞争条件的另一种情况是给定的一组指令以不正确的顺序被处理。竞争条件会导致底层系统出现不可预测的问题。它们通常很难检测和调试。

Python 的词汇表页面直接提到了 LBYL 编码风格引入了竞态条件的风险:

> 在多线程环境中，LBYL 方法可能会在“寻找”和“跳跃”之间引入竞争条件。例如，如果另一个线程在测试之后、查找之前从`mapping`中移除了`key`，那么代码`if key in mapping: return mapping[key]`可能会失败。这个问题可以通过锁或使用 EAFP 方法来解决。([来源](https://docs.python.org/3/glossary.html#term-LBYL))

竞争条件的风险不仅适用于多线程环境，也适用于 Python 编程中的其他常见情况。

例如，假设您已经设置了一个到正在使用的数据库的连接。现在，为了防止可能损坏数据库的问题，您需要检查连接是否处于活动状态:

```py
connection = create_connection(db, host, user, password)

# Later in your code...
if connection.is_active():
    # Update your database here...
    connection.commit()
else:
    # Handle the connection error here...
```

如果数据库主机在调用`.is_active()`和执行`if`代码块之间变得不可用，那么您的代码将会失败，因为主机不可用。

为了防止这种失败的风险，您可以使用 EAFP 编码风格，做一些类似这样的事情:

```py
connection = create_connection(db, host, user, password)

# Later in your code...
try:
    # Update your database here...
    connection.commit()
except ConnectionError:
    # Handle the connection error here...
```

这段代码继续尝试更新数据库，而不检查连接是否是活动的，这消除了检查和实际操作之间发生竞争的风险。如果出现`ConnectionError`，那么`except`代码块会适当地处理错误。这种方法会产生更健壮、更可靠的代码，将您从难以调试的竞争环境中解救出来。

### 提高代码的性能

在使用 LBYL 或 EAFP 时，性能是一个重要的考虑因素。如果您来自一种具有昂贵的异常处理过程的编程语言，那么这种担心是完全可以理解的。

然而，大多数 Python [实现](https://www.python.org/download/alternatives/)都努力让异常处理成为一种廉价的操作。所以，当你写 Python 代码的时候，你不应该担心异常的代价。在许多情况下，异常比条件语句更快。

根据经验，如果您的代码处理许多错误和异常情况，那么 LBYL 可以更高效，因为检查许多条件比处理许多异常的成本更低。

相比之下，如果你的代码只面临一些错误，那么 EAFP 可能是最有效的策略。在这些情况下，EAFP 会比 LBYL 快，因为你不会处理很多异常。您只需执行所需的操作，而无需一直检查先决条件的额外开销。

作为使用 LBYL 或 EAFP 如何影响代码性能的一个例子，假设您需要创建一个函数来测量给定文本中字符的频率。所以，你最终写下了:

>>>

```py
>>> def char_frequency_lbyl(text):
...     counter = {}
...     for char in text:
...         if char in counter:
...             counter[char] += 1
...         else:
...             counter[char] = 1
...     return counter
...
```

该函数将一段文本作为参数，并返回一个以字符为键的[字典](https://realpython.com/python-dicts/)。每个对应的值代表该字符在文本中出现的次数。

**注意:**在本教程的[提高代码性能](#improving-your-codes-performance)部分，您会发现一个补充本部分的例子。

为了构建这个字典， [`for`循环](https://realpython.com/python-for-loop/)遍历输入文本中的每个字符。在每次迭代中，条件语句检查当前字符是否已经在`counter`字典中。如果是这种情况，那么`if`代码块将字符的计数增加`1`。

另一方面，如果该字符还不在`counter`中，那么`else`代码块将该字符添加为一个键，并将其计数或频率设置为初始值`1`。最后，函数返回`counter`字典。

如果您使用一些示例文本调用您的函数，那么您将得到如下所示的结果:

>>>

```py
>>> sample_text = """
... Lorem ipsum dolor sit amet consectetur adipisicing elit. Maxime
... mollitia, molestiae quas vel sint commodi repudiandae consequuntur
... voluptatum laborum numquam blanditiis harum quisquam eius sed odit
... fugiat iusto fuga praesentium optio, eaque rerum! Provident similique
... accusantium nemo autem. Veritatis obcaecati tenetur iure eius earum
... ut molestias architecto voluptate aliquam nihil, eveniet aliquid
... culpa officia aut! Impedit sit sunt quaerat, odit, tenetur error,
... harum nesciunt ipsum debitis quas aliquid.
... """

>>> char_frequency_lbyl(sample_text)
{'\n': 9, 'L': 1, 'o': 24, 'r': 22, ..., 'V': 1, 'I': 1}
```

以`sample_text`作为参数调用`char_frequency_lbyl()`会返回一个包含字符计数对的字典。

如果你稍微思考一下寻找文本中字符频率的问题，那么你会意识到你需要考虑的字符数量是有限的。现在想想这个事实会如何影响你的解决方案。拥有有限数量的字符意味着你要做很多不必要的检查，看看当前字符是否已经在计数器中。

**注意:** Python 在 [`collections`](https://realpython.com/python-collections-module/) 模块中有一个专门的`Counter`类，用来处理计数对象的问题。查看 [Python 的计数器:Python 计数对象的方式](https://realpython.com/python-counter/)以获得更多细节。

一旦函数处理了一些文本，那么当您执行检查时，目标字符很可能已经在`counter`中了。最终，所有这些不必要的检查都会增加代码的性能成本。如果您正在处理大块文本，这一点尤其正确。

如何在代码中避免这种额外的开销？这时候 EAFP 就派上用场了。回到您的[交互式](https://realpython.com/interacting-with-python/)会话，编写以下函数:

>>>

```py
>>> def char_frequency_eafp(text):
...     counter = {}
...     for char in text:
...         try:
...             counter[char] += 1
...         except KeyError:
...             counter[char] = 1
...     return counter
...

>>> char_frequency_eafp(sample_text)
{'\n': 9, 'L': 1, 'o': 24, 'r': 22, ..., 'V': 1, 'I': 1}
```

这个函数的作用与前面例子中的`char_frequency_lbyl()`相同。然而，这一次，该函数使用 EAFP 编码风格。

现在，您可以对这两个函数运行一个快速的 [`timeit`](https://realpython.com/python-timer/#estimating-running-time-with-timeit) 性能测试，以了解哪一个更快:

>>>

```py
>>> import timeit
>>> sample_text *= 100

>>> eafp_time = min(
...     timeit.repeat(
...         stmt="char_frequency_eafp(sample_text)",
...         number=1000,
...         repeat=5,
...         globals=globals(),
...     )
... )

>>> lbyl_time = min(
...     timeit.repeat(
...         stmt="char_frequency_lbyl(sample_text)",
...         number=1000,
...         repeat=5,
...         globals=globals(),
...     )
... )

>>> print(f"LBYL is {lbyl_time / eafp_time:.3f} times slower than EAFP")
LBYL is 1.211 times slower than EAFP
```

在这个例子中，函数之间的性能差异很小。你可能会说这两种功能的表现是一样的。然而，随着文本大小的增长，函数之间的性能差异也成比例地增长，EAFP 实现最终比 LBYL 实现的效率略高。

从这个性能测试中得出的结论是，您需要事先考虑您要处理哪种问题。

你输入的数据大部分是正确的还是有效的？你只是在处理一些错误吗？就时间而言，你的先决条件代价大吗？如果你对这些问题的答案是*是*，那么请向 EAFP 倾斜。相比之下，如果你的数据很糟糕，你预计会发生很多错误，你的前提条件很轻，那么支持 LBYL。

[*Remove ads*](/account/join/)

### 总结:LBYL 对 EAFP

哇！您已经了解了很多关于 Python 的 LBYL 和 EAFP 编码风格。现在你知道这些风格是什么，它们的权衡是什么。要总结本节的主要主题和要点，请查看下表:

| 标准 | LBYL | EAFP |
| --- | --- | --- |
| 支票数量 | 重复通常由 Python 提供的检查 | 仅运行一次 Python 提供的检查 |
| 可读性和清晰性 | 可读性和清晰性较差，因为异常情况似乎比目标操作本身更重要 | 增强了可读性，因为目标操作位于前端和中心，而异常情况被置于后台 |
| 竞争条件风险 | 意味着检查和目标操作之间存在竞争条件的风险 | 防止竞争情况的风险，因为操作运行时不做任何检查 |
| 代码性能 | 当检查几乎总是成功时性能较差，而当检查几乎总是失败时性能较好 | 当检查几乎总是成功时，性能较好；当检查几乎总是失败时，性能较差 |

既然您已经深入比较了 LBYL 和 EAFP，那么是时候了解两种编码风格的一些常见问题以及如何在您的代码中避免它们了。连同上表中总结的主题，这些问题可以帮助您决定在给定的情况下使用哪种风格。

## LBYL 和 EAFP 的常见问题

当你使用 LBYL 风格编写代码时，你必须意识到你可能忽略了某些需要检查的条件。为了澄清这一点，回到将字符串值转换为整数的示例:

>>>

```py
>>> value = "42"

>>> if value.isdigit():
...     number = int(value)
... else:
...     number = 0
...

>>> number
42
```

显然，`.isdigit()`支票满足了你的所有需求。但是，如果您必须处理一个表示负数的字符串，该怎么办呢？`.isdigit()`对你有用吗？使用有效的负数作为字符串运行上面的示例，并检查会发生什么情况:

>>>

```py
>>> value = "-42"

>>> if value.isdigit():
...     number = int(value)
... else:
...     number = 0
...

>>> number
0
```

现在你得到的是`0`而不是预期的`-42`数。刚刚发生了什么？嗯，`.isdigit()`只检查从`0`到`9`的数字。它不检查负数。这种行为使得您的检查对于您的新需求来说是不完整的。你在检查前提条件时忽略了负数。

您也可以考虑使用`.isnumeric()`，但是这个方法也不会返回带有负值的`True`:

>>>

```py
>>> value = "-42"

>>> if value.isnumeric():
...     number = int(value)
... else:
...     number = 0
...

>>> number
0
```

这张支票不能满足你的需要。你需要尝试一些不同的东西。现在考虑如何防止在这个例子中遗漏必要检查的风险。是的，你可以使用 EAFP 编码风格:

>>>

```py
>>> value = "-42"

>>> try:
...     number = int(value)
... except ValueError:
...     number = 0
...

>>> number
-42
```

酷！现在，您的代码按预期工作。它转换正值和负值。为什么？因为默认情况下，将字符串转换为整数所需的所有条件都隐式包含在对`int()`的调用中。

到目前为止，看起来 EAFP 是你所有问题的答案。然而，事实并非总是如此。这种风格也有它的缺点。特别是，不能运行有副作用的代码。

考虑以下示例，该示例将问候消息写入文本文件:

```py
moments = ["morning", "afternoon", "evening"]
index = 3

with open("hello.txt", mode="w", encoding="utf-8") as hello:
    try:
 hello.write("Good\n") hello.write(f"{moments[index]}!")    except IndexError:
        pass
```

在本例中，您有一个表示一天中不同时刻的字符串列表。然后你用 [`with`语句](https://realpython.com/python-with-statement/)以写模式`"w"`打开`hello.txt` [文件](https://realpython.com/working-with-files-in-python/)。

`try`代码块包括对 [`.write()`](https://docs.python.org/3/library/io.html#io.TextIOBase.write) 的两次调用。第一个将问候语的开始部分写入目标文件。第二个调用通过从列表中检索一个时刻并将其写入文件来完成问候。

在第二次调用`.write()`期间，`except`语句捕获任何`IndexError`，后者执行索引操作以获得适当的参数。如果索引超出范围，就像示例中那样，那么您会得到一个`IndexError`，而`except`块会消除错误。然而，对`.write()`的第一次调用已经将`"Good\n"`写入了`hello.txt`文件，这最终导致了一种不期望的状态。

这种副作用在某些情况下可能很难恢复，所以您最好避免这样做。要解决此问题，您可以这样做:

```py
moments = ["morning", "afternoon", "evening"]
index = 3

with open("hello.txt", mode="w", encoding="utf-8") as hello:
    try:
 moment = f"{moments[index]}!"    except IndexError:
        pass
    else:
 hello.write("Good\n") hello.write(moment)
```

这一次，`try`代码块只运行索引，也就是可以引发一个`IndexError`的操作。如果出现这样的错误，那么您只需在`except`代码块中忽略它，让文件为空。如果索引成功，那么在`else`代码块中向文件写入完整的问候。

继续用`index = 0`和`index = 3`测试这两个代码片段，看看你的`hello.txt`文件会发生什么。

当你在`except`语句中使用一个宽泛的异常类时，EAFP 的第二个陷阱就出现了。例如，如果您正在处理一段可能引发几种异常类型的代码，那么您可能会考虑在`except`语句中使用`Exception`类，或者更糟，根本不使用异常类。

为什么这种做法是一个问题？嗯， [`Exception`](https://docs.python.org/3/library/exceptions.html#Exception) 类是几乎所有 Python 内置异常的父类。所以，你几乎可以捕捉到代码中的任何东西。结论是，您不会清楚地知道在给定的时刻您正在处理哪个错误或异常。

在实践中，避免做这样的事情:

```py
try:
    do_something()
except Exception:
    pass
```

显然，您的`do_something()`函数可以引发多种类型的异常。在所有情况下，您只需消除错误并继续执行您的程序。消除所有的错误，包括未知的错误，可能会导致以后意想不到的错误，这违反了 Python 的[法则中的一个基本原则:*错误永远不应该被忽略。*](https://peps.python.org/pep-0020/)

为了避免将来的麻烦，尽可能使用具体的异常。使用那些您有意识地期望代码引发的异常。记住你可以有几个`except`分支。例如，假设您已经测试了`do_something()`函数，并且您希望它引发`ValueError`和`IndexError`异常。在这种情况下，您可以这样做:

```py
try:
    do_something()
except ValueError:
    # Handle the ValueError here...
except IndexError:
    # Handle the IndexError here...
```

在这个例子中，拥有多个`except`分支允许您适当地处理每个预期的异常。这种构造还有一个优点，就是使您的代码更容易调试。为什么？因为如果`do_something()`引发意外异常，您的代码将立即失败。这样，您可以防止未知错误无声无息地传递。

[*Remove ads*](/account/join/)

## EAFP vs LBYL 举例

到目前为止，您已经了解了什么是 LBYL 和 EAFP，它们是如何工作的，以及这两种编码风格的优缺点。在这一节中，您将更深入地了解何时使用这种或那种样式。为此，您将编写一些实际的例子。

在开始举例之前，这里有一个何时使用 LBYL 或 EAFP 的总结:

| 将 LBYL 用于 | 将 EAFP 用于 |
| --- | --- |
| 可能失败的操作 | 不太可能失败的操作 |
| 不可撤销的手术，以及可能有副作用的手术 | 输入和输出(IO)操作，主要是硬盘和网络操作 |
| 可以提前快速预防的常见异常情况 | 可以快速回滚的数据库操作 |

有了这个总结，您就可以开始使用 LBYL 和 EAFP 编写一些实际的例子，展示这两种编码风格在现实编程中的优缺点。

### 处理过多的错误或异常情况

如果你预计你的代码会遇到大量的错误和异常情况，那么考虑使用 LBYL 而不是 EAFP。在这种情况下，LBYL 会更安全，可能会有更好的表现。

例如，假设您想编写一个函数来计算一段文本中单词的频率。要做到这一点，你计划使用字典。键将保存单词，值将存储它们的计数或频率。

因为自然语言有太多可能的单词需要考虑，所以你的代码将会处理许多`KeyError`异常。尽管如此，您还是决定使用 EAFP 编码风格。您最终会得到以下函数:

>>>

```py
>>> def word_frequency_eafp(text):
...     counter = {}
...     for word in text.split():
...         try:
...             counter[word] += 1
...         except KeyError:
...             counter[word] = 1
...     return counter
...

>>> sample_text = """
... Lorem ipsum dolor sit amet consectetur adipisicing elit. Maxime
... mollitia, molestiae quas vel sint commodi repudiandae consequuntur
... voluptatum laborum numquam blanditiis harum quisquam eius sed odit
... fugiat iusto fuga praesentium optio, eaque rerum! Provident similique
... accusantium nemo autem. Veritatis obcaecati tenetur iure eius earum
... ut molestias architecto voluptate aliquam nihil, eveniet aliquid
... culpa officia aut! Impedit sit sunt quaerat, odit, tenetur error,
... harum nesciunt ipsum debitis quas aliquid.
... """

>>> word_frequency_eafp(sample_text)
{'Lorem': 1, 'ipsum': 2, 'dolor': 1, ..., 'aliquid.': 1}
```

这个函数创建一个`counter`字典来存储单词和它们的计数。`for`循环遍历输入文本中的单词。在`try`块中，您试图通过将`1`加到当前单词的上一个值来更新当前单词的计数。如果目标单词在`counter`中不作为关键字存在，那么这个操作会引发一个`KeyError`。

`except`语句捕获`KeyError`异常，并用值`1`初始化`counter`中丢失的键——一个单词。

当您使用一些示例文本调用函数时，您会得到一个字典，其中单词作为键，计数作为值。就是这样！你解决了问题！

你的函数看起来不错！你用的是 EAFP 风格，而且很有效。但是，该函数可能比 LBYL 函数慢:

>>>

```py
>>> def word_frequency_lbyl(text):
...     counter = {}
...     for word in text.split():
...         if word in counter:
...             counter[word] += 1
...         else:
...             counter[word] = 1
...     return counter
...

>>> word_frequency_lbyl(sample_text)
{'Lorem': 1, 'ipsum': 2, 'dolor': 1, ..., 'aliquid.': 1}
```

在这个变体中，您使用一个条件语句来预先检查当前单词是否已经存在于`counter`字典中。如果是这种情况，那么您将计数增加`1`。否则，创建相应的键并将其值初始化为`1`。当您对示例文本运行该函数时，您会得到相同的字数对字典。

这个基于 LBYL 的实现与基于 EAFP 的实现获得相同的结果。但是，它可以有更好的性能。要确认这种可能性，请继续运行以下性能测试:

>>>

```py
>>> import timeit

>>> lbyl_time = min(
...     timeit.repeat(
...         stmt="word_frequency_lbyl(sample_text)",
...         number=1000,
...         repeat=5,
...         globals=globals(),
...     )
... )

>>> eafp_time = min(
...     timeit.repeat(
...         stmt="word_frequency_eafp(sample_text)",
...         number=1000,
...         repeat=5,
...         globals=globals(),
...     )
... )

>>> print(f"EAFP is {eafp_time / lbyl_time:.3f} times slower than LBYL")
EAFP is 2.117 times slower than LBYL
```

EAFP 并不总是你所有问题的最佳解决方案。在这个例子中，EAFP 比 LBYL 慢两倍多。

所以，如果错误和异常情况在你的代码中很常见，那就选择 LBYL 而不是 EAFP。许多条件语句可能比许多异常更快，因为在 Python 中检查条件仍然比处理异常成本更低。

[*Remove ads*](/account/join/)

### 检查对象的类型和属性

在 Python 中，检查对象的类型被广泛认为是一种反模式，应该尽可能避免。一些 Python 核心开发人员明确地将这种实践称为反模式，他们说:

> […]目前，Python 代码的一种常见反模式是检查接收到的参数的类型，以决定如何处理对象。
> 
> 【这种编码模式是】“脆弱且对扩展封闭”([来源](https://www.python.org/dev/peps/pep-0443/))。

使用类型检查反模式至少会影响 Python 编码的两个核心原则:

1.  [多态性](https://en.wikipedia.org/wiki/Polymorphism_(computer_science))，即单个接口可以处理不同类的对象
2.  [鸭式分类](https://realpython.com/python-type-checking/#duck-typing)，这是指一个物体具有决定它是否可以用于给定目的的特征

Python 通常依赖于对象的行为，而不是类型。例如，您应该有一个使用 [`.append()`](https://realpython.com/python-append/) 方法的函数。相反，你不应该有一个期待一个`list`参数的函数。为什么？因为将函数的行为绑定到参数的类型会牺牲鸭类型。

考虑以下函数:

```py
def add_users(username, users):
    if isinstance(users, list):
        users.append(username)
```

这个功能工作正常。它接受用户名和用户列表，并将新用户名添加到列表的末尾。然而，这个函数没有利用 duck 类型，因为它依赖于其参数的类型，而不是所需的行为，后者有一个`.append()`方法。

例如，如果您决定使用一个 [`collections.deque()`](https://realpython.com/python-deque/) 对象来存储您的`users`列表，那么如果您想让您的代码继续工作，您就必须修改这个函数。

为了避免用类型检查牺牲鸭式键入，您可以使用 EAFP 编码风格:

```py
def add_user(username, users):
    try:
        users.append(username)
    except AttributeError:
        pass
```

`add_user()`的实现不依赖于`users`的类型，而是依赖于它的`.append()`行为。有了这个新的实现，您可以立即开始使用一个`deque`对象来存储您的用户列表，或者您可以继续使用一个`list`对象。你不需要修改函数来保持你的代码工作。

Python 通常通过直接调用对象的方法和访问对象的属性来与对象进行交互，而无需事先检查对象的类型。在这些情况下，EAFP 编码风格是正确的选择。

影响多态性和 duck 类型的一个实践是，在代码中访问一个对象之前，检查它是否具有某些属性。考虑下面的例子:

```py
def get_user_roles(user):
    if hasattr(user, "roles"):
        return user.roles
    return None
```

在这个例子中，`get_user_roles()`使用 LBYL 编码风格来检查`user`对象是否有一个`.roles`属性。如果是这样，那么函数返回`.roles`的内容。否则，函数返回`None`。

不用通过使用内置的 [`hasattr()`](https://docs.python.org/3/library/functions.html#hasattr) 函数来检查`user`是否有一个`.roles`属性，您应该直接用 EAFP 风格访问该属性:

```py
def get_user_roles(user):
    try:
        return user.roles
    except AttributeError:
        return None
```

`get_user_roles()`的这个变体更加明确、直接和简单。它比基于 LBYL 的变体更具 Pythonic 风格。最后，它也可以更有效，因为它不是通过调用`hasattr()`不断检查前提条件。

[*Remove ads*](/account/join/)

### 使用文件和目录

管理文件系统上的文件和目录有时是 Python 应用程序和[项目](https://realpython.com/intermediate-python-project-ideas/)中的一项需求。当涉及到处理文件和目录时，很多事情都可能出错。

例如，假设您需要打开文件系统中的一个给定文件。如果你使用 LBYL 编码风格，那么你可以得到这样一段代码:

```py
from pathlib import Path

file_path = Path("/path/to/file.txt")

if file_path.exists():
    with file_path.open() as file:
        print(file.read())
else:
    print("file not found")
```

如果您针对文件系统中的一个文件运行这段代码，那么您将在屏幕上打印出该文件的内容。所以，这段代码有效。然而，它有一个隐藏的问题。如果由于某种原因，您的文件在检查文件是否存在和尝试打开它之间被删除，那么文件打开操作将失败并出现错误，您的代码将崩溃。

你如何避免这种竞争情况？你可以使用 EAFP 编码风格，如下面的代码所示:

```py
from pathlib import Path

file_path = Path("/path/to/file.txt")

try:
    with file_path.open() as file:
        print(file.read())
except IOError as e:
    print("file not found")
```

你不是检查你是否能打开文件，而是试图打开它。如果这行得通，那就太好了！如果它不起作用，那么您可以捕获错误并适当地处理它。请注意，您不再冒陷入竞争状态的风险。你现在安全了。

## 结论

现在你知道 Python 有**三思而后行** (LBYL)和**请求原谅比请求许可更容易** (EAFP)的编码风格，这是处理代码中的错误和异常情况的一般策略。您还了解了这些编码风格是什么，以及如何在代码中使用它们。

**在本教程中，您已经学习了:**

*   Python 的 **LBYL** 和 **EAFP** 编码风格的基础
*   Python 中 LBYL vs EAFP 的**利弊**和**利弊**
*   决定何时使用 LBYL 或 EAFP 的关键

有了关于 Python 的 LBYL 和 EAFP 编码风格的知识，您现在就能够决定在处理代码中的错误和异常情况时使用哪种策略。

**免费奖励:** [掌握 Python 的 5 个想法](https://realpython.com/bonus/python-mastery-course/)，这是一个面向 Python 开发者的免费课程，向您展示将 Python 技能提升到下一个水平所需的路线图和心态。*******