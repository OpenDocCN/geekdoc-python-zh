# Python 中的字符串格式

> 原文：<https://www.pythonforbeginners.com/basics/strings-formatting>

字符串格式化是一种将变量或另一个字符串插入预定义字符串的方法。每当我们需要向字符串中插入用户输入时，我们都可以使用字符串格式。在本文中，我们将在 python 中实现字符串格式化，并使用不同的示例来理解它。

## python 中的字符串格式化是什么？

字符串格式化是将变量值插入到字符串中的过程，以使程序员能够在不需要执行字符串连接的情况下向字符串添加新值。

如果用户输入他的年龄，程序需要打印句子“用户 x 岁”其中 x 是作为输入给出的用户年龄。在这种情况下，为了打印输出句子，我们使用如下的[字符串连接](https://www.pythonforbeginners.com/concatenation/string-concatenation-and-formatting-in-python)。

```py
myInput = input("Input your age:")
try:
    age = int(myInput)
    print("User is "+str(age)+" years old.")
except:
    print("Input is not in correct format")
```

输出:

```py
Input your age:12
User is 12 years old.
```

通过使用字符串格式化，可以实现上述程序，从而在不使用字符串连接的情况下给出相同的结果。

## 如何执行字符串格式化？

为了在 python 中执行字符串格式化，我们使用 format()方法。

首先，我们用花括号{}将一些占位符放在字符串中要放置输入变量的位置。当在包含占位符的字符串上调用 format()方法时，它接受与字符串中占位符数量相等的输入参数，并以连续的方式将输入放在占位符的位置。

```py
myInput = input("Input your age:")
try:
    age = int(myInput)
    print("User is {} years old.".format(age))
except:
    print("Input is not in correct format")
```

输出:

```py
Input your age:12
User is 12 years old.
```

我们也可以给 format()方法多个参数。当多个参数作为输入传递给方法时，它会用相应的参数替换占位符。

例如，假设用户将他/她的姓名和年龄作为输入输入到程序中，我们必须打印一个句子“用户 y 是 x 岁。”其中 y 是用户的姓名，x 是用户的年龄。我们将这样做。

```py
myName=input("Input your name:")
myInput = input("Input your age:")
try:
    age = int(myInput)
    print("User {} is {} years old.".format(myName, age))
except:
    print("Input is not in correct format")
```

输出:

```py
Input your name:Aditya
Input your age:22
User Aditya is 22 years old.
```

我们也可以给占位符分配索引。通过给占位符分配索引，我们可以按一定的顺序提供输入参数，相应索引处的大括号将被输入参数替换。

在这种方法中，输入不会连续映射到占位符。它们根据索引进行映射。这可以从下面的程序中理解。

```py
myName=input("Input your name:")
myInput = input("Input your age:")
try:
    age = int(myInput)
    print("User {1} is {0} years old.".format(age,myName))
except:
    print("Input age is not in correct format")
```

输出:

```py
Input your name:Aditya
Input your age:22
User Aditya is 22 years old.
```

format()方法也接受关键字参数。为了将输入作为关键字参数传递给 format 方法，我们首先命名占位符。随后，我们通过使用占位符名称作为关键字，将输入参数传递给方法。这可以从下面的例子中看出。

```py
myName=input("Input your name:")
myInput = input("Input your age:")
try:
    myAge = int(myInput)
    print("User {name} is {age} years old.".format(age=myAge,name=myName))
except:
    print("Input is not in correct format")
```

输出:

```py
Input your name:Aditya
Input your age:22
User Aditya is 22 years old.
```

使用占位符和 format()方法，我们还可以格式化程序中使用的输入浮点数，如下所示。

```py
myInput = input("Input a number:")
try:
    myNum = float(myInput)
    print("Input number is: {:.2f}".format(myNum))
except:
    print("Input is not in correct format")

```

输出:

```py
Input a number:12.2345
Input number is: 12.23
```

## 结论

在本文中，我们研究了使用 format()方法实现不同类型的字符串格式..我们必须使用 [python try except](https://www.pythonforbeginners.com/error-handling/python-try-and-except) 编写本文中使用的异常处理程序，以使程序更加健壮，并以系统的方式处理错误，以便在将输入从字符串转换为整数或浮点时不会出现错误。请继续关注更多内容丰富的文章。请继续关注更多内容丰富的文章。

## Yoast SEO