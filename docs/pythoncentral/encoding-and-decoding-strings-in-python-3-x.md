# 编码和解码字符串(在 Python 3.x 中)

> 原文：<https://www.pythoncentral.io/encoding-and-decoding-strings-in-python-3-x/>

在我们的另一篇文章[编码和解码字符串(在 Python 2.x 中)](https://www.pythoncentral.io/python-unicode-encode-decode-strings-python-2x/ "Encoding and Decoding Strings (in Python 2.x)")中，我们研究了 Python 2.x 如何处理字符串编码。在这里，我们将看看 Python 3.x 中的字符串编码和解码，以及有何不同。

## **Python 3 . x 与 Python 2.x 中的字符串编码/解码**

当 Python 2.x 发展到最新的 Python 3.x 版本时，该语言的许多方面并没有发生很大的变化。Python 字符串是*而不是*其中之一，事实上它可能是变化最大的。与 Python 2.x 相比，它所经历的变化在 Python 3.x 的编码/解码中处理字符串的方式上最为明显。在 Python 2.x 中编码和解码字符串有点麻烦，您可能在另一篇文章中读到过。令人欣慰的是，将 8 位字符串转换为 unicode 字符串，反之亦然，在 Python 3.x 中，两者之间的所有方法都被遗忘了。让我们直接通过一些示例来检查这意味着什么。

我们将从一个包含非 ASCII 字符(即“ü”或“umlaut-u”)的示例字符串开始:

```py

s = 'Flügel'

```

现在，如果我们引用并打印该字符串，它会给出基本相同的结果:

```py

>>> s

'Flügel'

>>> print(s)

Flügel

```

与 Python 2.x 中的相同字符串`s`相比，在这种情况下`s`已经是 Unicode 字符串，而*在 Python 3.x 中所有的*字符串都是自动 Unicode 的。明显的区别是*的*在我们实例化后没有改变。

虽然我们的字符串值包含一个非 ASCII 字符，但它离 ASCII 字符集不远，也就是基本拉丁字符集(实际上它是基本拉丁字符集的一部分)。如果我们有一个字符，不仅是非 ASCII 字符，而且是非拉丁字符，会发生什么？让我们来试试:

```py

>>> nonlat = '字'

>>> nonlat

'字'

>>> print(nonlat)

字

```

正如我们所看到的，它是否包含所有拉丁字符并不重要，因为 Python 3.x 中的字符串都是这样的(与 Python 2.x 不同，您可以在空闲窗口中键入任何字符！).

如果你在 Python 2.x 中处理过[编码和解码字符串，那么](https://www.pythoncentral.io/python-unicode-encode-decode-strings-python-2x/)你会知道处理起来会麻烦得多，而 Python 3.x 让这变得不那么痛苦了。然而，如果我们不需要使用`unicode`、`encode`或`decode`方法，或者在我们的字符串变量中包含多个反斜杠转义来立即使用它们，那么我们还有什么必要编码或解码我们的 Python 3.x 字符串呢？在回答这个问题之前，我们先来看一下 Python 3.x 中的`b'...'`(字节)对象，与 Python 2.x 中的对象形成对比。

### **Python 3 . x Bytes 对象**

在 Python 2.x 中，在字符串前面加上“B”(或“B”)是合法的语法，但它没有什么特别之处:

```py

>>> b'prefix in Python 2.x'

'prefix in Python 2.x'

```

然而，在 Python 3.x 中，这个前缀表示字符串是一个不同于普通字符串的`bytes`对象(我们知道普通字符串默认为 Unicode 字符串)，甚至“b”前缀也被保留:

```py

>>> b'prefix in Python 3.x'

b'prefix in Python 3.x'

```

关于字节对象的事情是，它们实际上是整数的数组，尽管我们把它们看作 ASCII 字符。在这一点上，它们如何或者为什么是整数数组对我们来说并不重要，但是重要的是我们将只把它们看作一串 ASCII 文字字符，并且它们可以*只有*包含 ASCII 文字字符。这就是为什么下面的代码(或任何非 ASCII 字符)不起作用的原因:

```py

>>> b'字'

SyntaxError: bytes can only contain ASCII literal characters.

```

现在，为了了解字节对象与字符串的关系，我们先来看看如何将字符串转换成字节对象，反之亦然。

### **将 Python 字符串转换为字节，并将字节转换为字符串**

如果我们想把之前的`nonlat`字符串转换成 bytes 对象，我们可以使用`bytes`构造函数方法；然而，如果我们只使用字符串作为唯一的参数，我们会得到这个错误:

```py

>>> bytes(nonlat)

Traceback (most recent call last):

File "<stdin>", line 1, in <module>

TypeError: string argument without an encoding

```

正如我们所看到的，我们需要在字符串中包含一个编码。让我们用一个常见的，UTF 8 编码:

```py

>>> bytes(nonlat, 'utf-8')

b'\xe5\xad\x97'

```

现在我们有了我们的`bytes`物体，用 UTF 8 编码...但是这到底是什么意思呢？这意味着包含在我们的`nonlat`变量中的单个字符被有效地翻译成一串代码，这意味着“字“在 UTF-8 中——换句话说，它是用*编码的*。这是否意味着如果我们在`nonlat`上使用`encode`方法调用，我们会得到相同的结果？让我们看看:

```py

>>> nonlat.encode()

b'\xe5\xad\x97'

```

事实上，我们得到了相同的结果，但在这种情况下我们不必给出编码，因为 Python 3.x 中的 *encode* 方法默认使用 UTF-8 编码。如果我们将其更改为 UTF-16，我们会得到不同的结果:

```py

>>> nonlat.encode('utf-16')

b'\xff\xfeW['

```

尽管这两个调用执行相同的功能，但它们根据编码或编解码器的不同，以稍微不同的方式执行。

既然我们可以对字符串进行编码以生成字节，我们也可以对字节进行解码以生成字符串——但是当解码一个字节对象时，我们*必须*知道使用正确的编解码器来获得正确的结果。例如，如果我们尝试使用 UTF-8 来解码上面 nonlat 的 UTF-16 编码版本:

```py

# We can use the method directly on the bytes

>>> b'\xff\xfeW['.decode('utf-8')

Traceback (most recent call last):

File "<stdin>", line 1, in <module>

UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte

```

我们得到一个错误！现在，如果我们使用正确的编解码器，结果会很好:

```py

>>> b'\xff\xfeW['.decode('utf-16')

'字'

```

在这种情况下，由于解码操作失败，Python 向我们发出了警告，但警告是，当编解码器不正确时，错误不会总是发生！这是因为编解码器通常使用相同的代码短语(组成 bytes 对象的“\xXXX”转义)来表示不同的内容！如果我们在人类语言的背景下考虑这一点，使用不同的编解码器编码和解码相同的信息就像试图用意大利语-英语词典将一个或多个单词从西班牙语翻译成英语一样——意大利语和西班牙语的一些音素可能相似，但你仍然会得到错误的翻译！

### **在 Python 3.x 中向文件写入非 ASCII 数据**

关于 Python 3.x 和 Python 2.x 中的字符串，最后一点要注意的是，我们必须记住，使用`open`方法写入两个分支中的文件不允许将 Unicode 字符串(包含非 ASCII 字符)写入文件。为了做到这一点，字符串必须经过*编码*。

这在 Python 2.x 中没什么大不了的，因为只有当你这样做时(通过使用`unicode`方法或`str.decode`)，字符串才会是 Unicode 的，但是在 Python 3.x 中，默认情况下所有字符串都是 Unicode 的，所以如果我们想将这样的字符串(例如`nonlat`)写入文件，我们需要使用`str.encode`和`open`的`wb`(二进制)模式将字符串写入文件，而不会导致错误，如下所示:

```py

>>> with open('nonlat.txt', 'wb') as f:

f.write(nonlat.encode())

```

同样，当读取非 ASCII 数据的文件时，使用`rb`模式并用*正确的*编解码器对数据进行*解码*也很重要——当然，除非你不介意用“意大利语”翻译你的“西班牙语”