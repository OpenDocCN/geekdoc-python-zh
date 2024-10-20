# Python 中的条件——快速指南

> 原文：<https://www.askpython.com/python/examples/conditionals-in-python>

这个世界比看起来要复杂得多。这就是为什么在某些情况下我们需要设置条件来解决复杂性。在条件句的帮助下，你可以跳过一些复杂的内容或者快速运行一系列的语句。您也可以在语句和执行程序之间进行选择。例如，在日常生活中，我们使用很多条件句，如如果今天下雪，我就不去市场了，否则我就去。

## Python 中的条件句

在本教程中，我们将学习如何在 Python 中使用条件句。所以让我们开始吧！

我们一个一个来看条件语句。

### 如果语句

我们将从最基本的“如果”陈述开始。if 语句的语法如下:

```py
if(condition):
             (statement)

```

语法的意思是，如果条件满足，语句将被执行，否则将被跳过。

**例如:**

```py
A=1
B=3

if(A<B):
        print(‘A is true’)

```

**输出:**

```py
A is true

```

**现在，如果我们想一起执行多条语句，if 语句看起来会像这样:**

```py
if(condition):

               <statement>
               <statement>
                <statement>
                 ……………

< Other statements>

```

### Else 语句

现在让我们看看 else 条件是什么样的。在这种情况下，我们将把 if 语句与 else 条件结合起来。

**else 语句的基本语法:**

```py
if(condition):
                <statement>

else:
              <statement>

```

**例如:**

```py
x=200

If x<300:
         print(‘200 is smaller’)

else:
        print(‘200 is bigger’)

```

**输出:**

```py
200 is smaller

```

### Elif 语句

elif 语句是 Python 中的另一个条件语句，有助于检查多个为真的条件。Elif 语句几乎类似于 if 语句，只是只能有一个 else 语句，而可以有多个 elif 语句。

下面是 elif 语句的语法。

```py
if(condition1):
              <statement>

elif(condition2):
                <statement>
elif(condition3):
                  <statement>
else:
          <statement>

```

让我们来看看下面的例子。

```py
price=180

if price>100:
                print(“Price is greater than 100”)

elif price==100:
                 print(“Price is 100”)

elif price<100:
                 print(“Price is less than 100”)

else :
                print(“Invalid Price”)

```

**输出:**

```py
Price is greater than 100

```

**注意:**所有的语句都应该保持它们的缩进级别，否则会抛出一个缩进错误。

## 结论:

在本文中，我们学习了 Python 中的条件或控制结构。这些控制结构决定了程序的执行。您可以使用循环和这些控制结构来执行不同的程序。因此，了解 Python 中的条件语句非常重要。