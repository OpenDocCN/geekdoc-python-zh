# Python 中的多条件 If 语句[解释]

> 原文：<https://www.askpython.com/python/examples/multi-conditional-if-statement>

你好，初学者！今天，我们将了解如何在' [if 语句](https://www.askpython.com/course/python-course-if-else-statement)中实现多个条件。在本文结束时，您将了解实现 if-else 条件的不同情况。让我们开始吧。

* * *

## Python 中的 if 语句是什么？

“If”语句是一个条件语句，用于检查特定表达式是否为真。程序控制首先检查用‘if’写的条件，如果条件证明为真，则执行 if 块。否则，程序控制转到 else 块并执行它。

**语法:**

```py
if(condition) :
    code block 1  
else :
    code block 2 

```

如果满足条件，则执行代码块 1。如果没有，则执行代码块 2。

我们一般都使用基本的 if 语句，即只有一个条件的 if 语句。当我们想要将一个变量与另一个变量进行比较，或者我们想要检查一个变量是否为真时，就会用到这个函数。例如:

```py
num1 = int(input("Enter a number:")

if( num1 % 2 == 0 ):
    print("The number is even!")
else:
    print("The number is odd!")

```

输出:

```py
Enter a number: 37
The number is odd!

```

## 如何在 if 语句中使用多个条件？

现在，我们将了解如何在 if 语句中使用多个条件。语法和示例解释如下:

**语法**:

```py
if ((cond1) AND/OR (cond2)) :
    code block 1
else :
    code block 2

```

可以在单个 if 语句中使用**和**或**或**或**这两个**来使用多个条件。

### 1.使用“and”的多个条件

当你希望所有的条件都得到满足时，就使用 AND 条件。看看下面的例子:

```py
age = int (input (" What is your age? "))
exp = int (input (" Enter your work experience in years: "))

if (age > 30 and age < 60) and (exp > 4):
    Print (" You are hired! ")
else:
    Print (" Sorry! you are not eligible :( ")

```

上面的代码使用 AND 条件，这意味着每个条件都必须为真。年龄必须在 30 至 60 岁之间，经验应该超过 4 年，然后只有你会被录用。

```py
Output:
What is your age?  32
Enter your work experience in years: 6
You are hired!

What is your age? 28
Enter your work experience in years: 5
Sorry! you are not eligible :(  

```

### 2.使用“或”的多个条件

当您希望至少满足一个条件时，可以使用 OR 条件。让我们来看一个例子:

```py
num1 = int(input("Enter any number : "))
rem = num1 % 10

if (rem == 0 ) or ( rem == 5 ) :
    print( "{} is divisible by 5 ".format(num1))
else :
    print(" {} is not divisible by 5".format(num1))

```

上面的代码检查输入的数字是否能被 5 整除。为此，它首先通过找出除以 10 时的余数(使用模 10)来找出数字的最后一位，如果余数等于 0 或 5，它打印出该数字可被 5 整除。如果不是，它打印出这个数不能被 5 整除。

```py
OUTPUT :

Enter any number : 90
90 is divisible by 5 

Enter any number : 27
27 is not divisible by 5 

Enter any number : 15
15 is divisible by 5 

```

## 结论

这就是我们如何在 if 语句中使用多个条件。请尝试不同的 if-else 条件组合，如果有任何问题，请随时提问！

谢谢大家！🙂