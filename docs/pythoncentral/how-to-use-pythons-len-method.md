# 如何使用 Python 的 len()方法

> 原文：<https://www.pythoncentral.io/how-to-use-pythons-len-method/>

Python 的 len()方法可以用来轻松找到字符串的长度。这是一种简单快捷的测量字符串长度(字符数)的方法，无需编写大量代码。使用 len()方法的语法相当简单，很难出错——即使是初学 Python 的程序员也应该能够掌握它。要了解其工作原理，请查看以下代码片段:

```py
str = "hey, I'm a string, and I have a lot of characters...cool!"
print = "String length:", len(str)
```

所以要找到字符串“str”的长度，需要通过 len()方法传递。你的答案是字符串中的字符数，包括标点和空格。字符串“str”中的字符数是 57，包括空格、逗号、句点、撇号和感叹号。因此，上述代码的输出如下所示:

```py
String length: 57
```

len()方法还会计算字符串中包含的数字、空格和标点符号，如下例所示:

```py
str = "I'm 26 years old."
print = "String length:", len(str)
```

上述代码的输出如下，其中 str 字符串有 17 个字符，包括空格、撇号、数字和句点:

```py
String length: 17
```

len()方法除了字符串名称之外不接受任何参数，所以这是唯一可以通过它传递的东西。这不是一个特别通用的方法，但是对于这个特定的目的来说，它确实很方便，并且在您的编码冒险中，您可能会遇到许多不同的原因，在这些原因中，您需要找到字符串的长度。您还可以在 if 语句中使用此方法，根据字符串包含(或不包含)的字符数来执行某些代码。