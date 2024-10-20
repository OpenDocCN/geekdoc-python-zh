# 捕捉 Python 异常——try/except/else 关键字

> 原文：<https://www.pythoncentral.io/catching-python-exceptions-the-try-except-else-keywords/>

通常，在编写 python 杰作时，在执行您精心设计的代码时，会出现某些问题。诸如丢失的文件或目录、空字符串、应该是字符串但在运行时实际上是数组的变量。

这些东西在 Python 中被称为*异常*。这就是`try`关键字的用途。

它允许执行嵌套在一个合适的块中的潜在中断代码。这个块将试图*捕捉*任何那些讨厌的异常，然后执行代码或者从错误中恢复，或者通知用户这个错误。

例如，此函数希望运行:

```py

def succeed(yn):

    if yn:

        return True

    else:

        raise Exception("I can't succeed!")

```

但是，据我们所知，它有可能引发一个异常。当一个异常被引发，并且它不在一个有适当异常处理的`try`块中时，它将停止代码的执行。

如果此代码以下列身份运行:

```py

>>> succeed(True)

True

```

会没事的，没人会知道有什么不同。

但是，如果它运行为

```py

>>> succeed(False)

Traceback (most recent call last):

  File "<stdin>", line 1, in <module>

  File "<stdin>", line 5, in succeed

Exception: I can't succeed!

```

我们得到这个可爱的消息，代码不能成功。只是没那么可爱。其实挺丑的。

那么我们能做什么呢？

这就是`try`块的用武之地！

因此，为了使上面的代码运行时友好，我们可以这样做:

```py

from random import randint
def handle _ exception(e):
print(e)
print('但是我可以安全！')
try:
success(randint(False，True))
Exception as e:
handle _ Exception(e)

```

我们刚刚做了什么？嗯，`randint`会在两个给定的输入之间选择一个随机整数。在这种情况下，`False`为 0，`True`为 1。所以`succeed`函数会随机引发一个异常。

现在除了例外。如果`succeed`函数引发异常，我们告诉 Python 只执行`handle_exception`*。*

 *因此，如果我们运行这段代码，如果成功，输出将是空的，如果失败:

```py

I can't succeed!

But I can be safe!

```

但是如果你想在成功后执行一段代码呢？你可以这样做

```py

def another_method_that_could_fail():

    fail = randint(False, True)
如果失败:
抛出 runtime error(‘我肯定失败了。’)
 else: 
 print("耶！我没有失败！")
try:
success(randint(False，True))
another _ method _ that _ could _ fail()
Exception as e:
handle _ Exception(e)

```

现在，我们看到如果`succeed`没有引发异常，那么`another_method_that_could_fail`就会运行！太神奇了！我们做到了！

但是等等！如果`another_method_that_could_fail`运行，它将再次运行`handle_exception`，我们希望打印不同的消息。可恶。

那我们该怎么办？嗯，我们可以向我们的`try`块添加另一个块，如下所示:

```py

def handle_runtime(re):

    pass
try:
success(randint(False，True))
another _ method _ that _ could _ fail()
Exception runtime error as re:
handle _ runtime(re)
Exception as e:
handle _ Exception(e)

```

好吧，那很好。但是现在，我们如何成功运行一段代码，没有例外？嗯，`try`区块还有一部分不太为人所知。这是关键字`else`。观察:

如果我们将之前的代码稍加修改:

```py

try:

    succeed(randint(False, True))

    another_method_that_could_fail()

except RuntimeError as re:

    handle_runtime(re)

except Exception as e:

    handle_exception(e)

else:

    print('Yes! No exceptions this time!')

```

然后，只有当我们成功时，才会打印出一条漂亮的消息。好吧，但是如果我们无论如何都需要一段代码来运行呢？对于关闭一个已经在`try`块中打开的文件有用吗？是啊！

从舞台左侧，输入`finally`关键字。它会如我们所愿。

在光荣的行动中观察它:

```py

try:

    succeed(randint(False, True))

    another_method_that_could_fail()

except RuntimeError as re:

    handle_runtime(re)

except Exception as e:

    handle_exception(e)

else:

    print('Unknown error occurred. Exiting.')

    exit(2)

finally:

    print("Finally! I'm done. I don't care if I failed or not. I'm DONE.")

```

瞧！我们找到了。现在我们将总是在`finally`部分打印消息。

这就是 try/except/else/finally 块。我希望我在你成为一名 Python 大师的过程中很好地教育了你。

Sayō nara，暂时的。*