# Python 中如何切片字符串？

> 原文：<https://www.askpython.com/python/string/slice-strings-in-python>

## 介绍

在本教程中，我们将学习如何在 Python 中分割字符串。

Python 支持字符串切片。它是根据用户定义的起始和结束索引，从给定的字符串创建一个新的子字符串。

## Python 中分割字符串的方法

如果你想在 Python 中分割字符串，就像下面这一行一样简单。

```py
res_s = s[ start_pos:end_pos:step ]

```

这里，

*   **res_s** 存储返回的子串，
*   s 是给定的字符串，
*   **start_pos** 是我们需要对字符串 s 进行切片的起始索引，
*   **end_pos** 是结束索引，在此之前切片操作将结束，
*   **步骤**是切片过程从开始位置到结束位置的步骤。

**注意**:以上三个参数都是可选的。默认情况下，`start_pos`设置为 **0** ，`end_pos`被认为等于字符串的长度，`step`设置为 **1** 。

现在让我们举一些例子来理解如何更好地在 Python 中切分字符串。

## Python 中的切片字符串–示例

可以用不同的方式对 Python 字符串进行切片。

通常，我们使用简单的索引来访问字符串元素(字符)，从 *0* 到 **n-1** (n 是字符串的长度)。因此，为了访问字符串`string1`的第**个**元素，我们可以简单地使用下面的代码。

```py
s1 = String1[0]

```

还是那句话，访问这些字符还有另外一种方法，就是使用**负索引**。负分度从 **-1** 到 **-n** 开始(n 为给定字符串的长度)。注意，反向索引是从字符串的另一端开始的。因此，这次要访问第一个字符，我们需要遵循下面给出的代码。

```py
s1 = String﻿1[-n]

```

现在让我们来看看使用上述概念分割字符串的一些方法。

### 1.Python 中带开始和结束的切片字符串

我们可以通过提及我们正在寻找的期望子串的开始和结束索引来容易地分割给定的串。看下面给出的例子，它解释了使用开始和结束索引的字符串切片，包括通常的和负的索引方法。

```py
#string slicing with two parameters
s = "Hello World!"

res1 = s[2:8]
res2 = s[-4:-1] #using negative indexing

print("Result1 = ",res1)
print("Result2 = ",res2)

```

**输出**:

```py
Result1 =  llo Wo
Result2 =  rld

```

这里，

*   我们初始化一个字符串，`s`为**“Hello World！”**，
*   首先，我们用起始索引 **2** 和结束索引 **8** 对给定的字符串进行切片。这意味着得到的子串将包含从 **s[2]** 到 **s[8-1]** 的字符，
*   类似地，对于下一个，结果子字符串应该包含从 **s[-4]** 到 **s[(-1)-1]** 的字符。

因此，我们的产出是合理的。

### 2.仅使用开头或结尾分割字符串

如前所述，字符串切片的三个参数都是可选的。因此，我们可以使用一个参数轻松完成我们的任务。看看下面的代码就清楚了。

```py
#string slicing with one parameter
s1= "Charlie"
s2="Jordan"

res1 = s1[2:] #default value of ending position is set to the length of string
res2 = s2[:4] #default value of starting position is set to 0

print("Result1 = ",res1)
print("Result2 = ",res2)

```

**输出**:

```py
Result1 =  arlie
Result2 =  Jord

```

这里，

*   我们首先初始化两个字符串， **s1** 和 **s2** ，
*   为了对它们进行切片，我们只对 s1 提到了 **start_pos** ，对 s2 只提到了 **end_pos** ，
*   因此，对于 **res1** ，它包含从索引 2(如前所述)到最后一个(默认设置为 n-1)的 s1 的子串。而对于 res2，指数的范围从 0 到 4(已提及)。

### 3.Python 中带步长参数的切片字符串

`step`值决定切片操作从一个索引到另一个索引的跳转。仔细看下面的例子。

```py
#string slicing with step parameter
s= "Python"
s1="Kotlin"

res = s[0:5:2]
res1 = s1[-1:-4:-2] #using negative parameters

print("Resultant sliced string = ",res)
print("Resultant sliced string(negative parameters) = ",res1)

```

**输出**:

```py
Resultant sliced string =  Pto
Resultant sliced string(negative parameters) =  nl

```

在上面的代码中，

*   我们初始化两个字符串 **s** 和 **s1** ，并尝试按照给定的起始和结束索引对它们进行切片，就像我们对第一个示例所做的那样，
*   但是这次我们提到了一个**步骤**值，在前面的例子中它被默认设置为 1，
*   对于 res，步长为 2 意味着，当遍历从索引 0 到 4 的子串时，每次索引都会增加值 2。即第一个字符是**s[0]**(‘P’)，子串中的下一个字符将是 **s[0+2]** 和 **s[2+2]** ，直到索引刚好小于 5。
*   对于下一个，即 **res1** ，提到的步骤是(-2)。因此，与前面的情况类似，子字符串中的字符将是 **s1[-1]** ，然后是 **s1[(-1)+(-2)]** 或 **s1[-3]** ，直到索引刚好小于(-4)。

### 4.在 Python 中使用切片反转字符串

通过使用 Python 中的负索引字符串切片，我们还可以反转字符串并将其存储在另一个变量中。为此，我们只需要提到一个大小为 **(-1)** 的`step`。

让我们看看下面的例子是如何工作的。

```py
#reversing string using string slicing
s= "AskPython"
rev_s = s[::-1] #reverse string stored into rev_s

print(rev_s)

```

**输出**:

```py
nohtyPksA

```

我们可以看到，字符串 s 被反转并存储到`rev_s`中。**注意**:在这种情况下，原来的弦也保持完好无损。

## 结论

因此，在本教程中，我们学习了字符串切片方法及其不同的形式。希望，读者对这个话题有一个清楚的了解。

关于这个话题的任何进一步的问题，请随意使用下面的评论。

## 参考

*   Python 切片字符串——日志开发帖子，
*   [分割字符串的方法？](https://stackoverflow.com/questions/1010961/ways-to-slice-a-string)–stack overflow 问题。