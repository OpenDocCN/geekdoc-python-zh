# python–连接字符串和整数

> 原文：<https://www.askpython.com/python/string/python-concatenate-string-and-int>

在 Python 中，我们通常使用`+`操作符来执行字符串连接。然而，正如我们所知,`+`操作符也用于将整数或浮点数相加。

那么，如果操作数两边都有一个*字符串*和一个*整型*，会发生什么呢？

由于 Python 是一种动态类型的语言，我们在编译期间不会遇到任何错误，但是，我们会得到一个运行时错误。(更具体地说，引发了一个**类型错误**异常)

以下片段证明了这一点:

```py
a = "Hello, I am in grade "

b = 12

print(a + b)

```

输出

```py
Traceback (most recent call last):
  File "concat.py", line 5, in <module>
    print(a + b)
TypeError: can only concatenate str (not "int") to str

```

因此，由于我们不能直接将一个整数和一个字符串连接起来，我们需要操作操作数，以便它们可以被连接起来。有多种方法可以做到这一点。

* * *

## 1.使用 str()

我们可以通过`str()`函数将整数转换成字符串。现在，新字符串可以与另一个字符串连接起来，以给出输出；

```py
print(a + str(b))

```

**输出**

```py
Hello, I am in grade 12

```

这是将整数转换为字符串的最常见方式。

但是，我们也可以使用其他方法。

## 2.使用格式()

```py
a = "Hello, I am in grade "

b = 12

print("{}{}".format(a, b))

```

输出和以前一样。

## 3.使用“%”格式说明符

```py
a = "Hello, I am in grade "

b = 12

print("%s%s" % (a, b))

```

虽然我们可以指定`a`和`b`都是字符串，但是我们也可以使用 C 风格的格式说明符(`%d`、`%s`)将一个整数和一个字符串连接起来。

```py
print('%s%d' % (a,b))

```

以上代码的输出保持不变。

## 4.使用 f 弦

我们可以在 Python 3.6 或更高版本上使用 Python f-strings 来连接一个整数和一个字符串。

```py
a = "Hello, I am in grade "

b = 12

print(f"{a}{b}")

```

## 5.使用 print()打印字符串

如果我们想直接打印连接的字符串，我们可以使用`print()`来完成连接。

```py
a = "Hello, I am in grade "
b = 12
print(a, b, sep="")

```

我们使用空字符串分隔符(`sep`)连接`a`和`b`，因为`print()`的默认分隔符是空格(" ")。

* * *

## 结论

在本文中，我们学习了如何使用各种方法将一个整数连接成一个字符串。

## 参考

*   [StackOverflow 关于 string 和 int 串联的问题](https://stackoverflow.com/questions/25675943/how-can-i-concatenate-str-and-int-objects)
*   JournalDev 关于字符串和整型连接的文章

* * *