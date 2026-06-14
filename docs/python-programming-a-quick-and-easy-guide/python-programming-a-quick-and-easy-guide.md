

# Python 编程

快速入门指南

Fatos Morina

© 2022 Fatos Morina

## 目录

- Python 简介
- Python 基础
- 数据类型与变量
- 控制流
- 函数
- 模块与包
- 面向对象编程
- 文件处理
- 错误处理
- 使用库
- 结论

## 目录

总结

## Python 简介

Python 是最受欢迎的编程语言之一，由 Guido van Rossum 于 1991 年创建。

根据 Guido van Rossum 的说法，Python 是一种：

> “高级编程语言，其核心设计哲学完全围绕代码可读性和一种允许程序员用几行代码表达概念的语法。”

它代表了你可以学习并使用的最简单的语言之一。它可用于软件开发、Web 开发的服务端、机器学习、数学以及你能想到的任何类型的脚本编写。

它的好处在于，它被许多公司和学术机构广泛使用和接受，使其成为一个非常好的选择，特别是如果你刚刚开始你的编程之旅。此外，它有一个庞大的开发者社区，他们使用它并愿意提供帮助。这个社区已经发布了许多你可以开始使用的开源库。他们还积极地不断改进它们。

它的语法与英语非常相似，使你更容易理解并相当直观地使用它。

Python 运行在解释器系统上，这意味着你不需要等待编译器编译代码然后执行它。相反，你可以快速构建原型。

它可以在不同的平台上运行，例如 Windows、Linux、Mac、Raspberry Pi 等。

## 基础知识

与其他语言相比，Python 特别重视缩进。

在其他编程语言中，空格和缩进仅用于使代码更具可读性和更美观，而在 Python 中，它们代表一个代码子块。

以下代码将无法工作，因为第二行没有正确缩进：

```
if 100 > 10:
print("100 is greater than 10")
```

为了使其正常工作，缩进应如下所示：

```
if 100 > 10:
    print("100 is greater than 10")
```

这可能看起来很难理解，但你可以使用代码编辑器来非常生动地突出显示此类语法错误。此外，你编写的 Python 代码越多，你就越容易将这种缩进视为第二天性。

## 注释

我们使用注释来指定程序应简单忽略且不被 Python 解释器执行的部分。这意味着写在注释中的所有内容都不会被考虑，你可以用任何你想要的语言编写，包括你自己的母语言。

我们可以在行首使用符号 # 来开始注释：

```
# This is a comment in Python
```

我们也可以使用三引号在多行中包含更长的注释：

```
"""
This is a comment in Python.
Here we are typing in another line and we are still inside the same comment block.

In the previous line we have a space, because it is allowed to have spaces inside comments.

Let us end this comment right here.
"""
```

## print() 方法

我们将使用 print() 方法，因为它帮助我们在控制台中看到结果。

你现在不需要知道它在幕后是如何工作的，甚至不需要知道什么是方法。只需将其视为一种在控制台中显示代码结果的方式。

## 运算符

### 算术运算符

尽管你可能从口袋里掏出手机进行一些计算，但从现在开始你也应该习惯在 Python 中实现一些算术运算符。

当我们想将两个数字相加时，我们使用加号，就像在数学中一样：

```python
print(50 + 4)  # 54
```

类似地，对于减法，我们使用减号：

```python
print(50 - 4)  # 46
```

对于乘法，我们使用星号：

```python
print(50 * 4)  # 200
```

进行除法时，我们使用正斜杠：

```python
print(50 / 4)  # 12.5
print(8 / 4)   # 2.0
```

这会产生浮点数。如果我们只想在除法时获得整数，也称为简单地进行整数除法，我们应该使用双斜杠：

```
print(50 // 4)  # 12
print(8 // 4)   # 2
```

我们还可以使用百分号 % 找到一个数除以另一个数的余数。

```
print(50 % 4)  # 2
```

这个操作在我们想检查一个数是奇数还是偶数时特别有用。如果一个数除以 2 的余数是 1，那么它就是奇数。否则，它就是偶数。
这是一个示例：

```
print(50 % 2)  # 0
# 由于余数是 0，这个数是偶数

print(51 % 2)  # 1
# 由于余数是 1，这个数是奇数
```

当我们想将一个数提升到特定幂时，我们应该使用双星号：

```
print(2 ** 3)  # 8
# 这是 2 * 2 * 2 的简写

print(5 ** 4)  # 625
# 这是 5 * 5 * 5 * 5 的简写
```

### 赋值运算符

这些运算符用于将值赋给变量。
当我们声明一个变量时，我们使用等号：

```
name = "Fatos"
age = 28
```

我们也可以在一行中声明多个变量：

```
name, age, location = "Fatos", 28, "Europe"
```

我们也可以使用这种方式在变量之间交换值，例如，假设我们有两个变量 a 和 b，并想交换它们的值。

一种逻辑方法是引入一个作为临时变量的第三个变量：

```
a, b = 1, 2

print(a) # 1
print(b) # 2

c = a
a = b
b = c

print(a) # 2
print(b) # 1
```

然而，我们可以通过以下方式在一行中完成：

```
a, b = 1, 2

print(a)  # 1
print(b)  # 2

b, a = a, b

print(a)  # 2
print(b)  # 1
```

我们还可以将赋值运算符与算术运算符合并。

让我们先看看如何对加法进行此操作。

假设我们有以下变量：

```
total_sum = 20

current_sum = 10
```

现在我们想将 `current_sum` 的值加到 `total_sum`。为此，我们应该写：

```
total_sum = total_sum + current_sum

print(total_sum)  # 30
```

这可能看起来不准确，因为右边不等于左边。然而，在这种情况下，我们只是在进行赋值，而不是比较等式的两边。

为了快速完成此操作，我们可以使用以下形式：

```
total_sum += current_sum

print(total_sum)  # 30
```

这等同于前面的语句。
类似地，我们可以对其他算术运算符执行相同的操作：

- 减法：

    ```
    result = 3
    number = 4

    result -= number  # 这等同于 result = result - number

    print(result)  # -1
    ```

- 乘法：

    ```
    product = 3
    number = 4

    product *= number  # 这等同于 product = product * number

    print(product)  # 12
    ```

- 除法：

    ```
    result = 8
    number = 4

    result /= number  # 这等同于 result = result / number

    print(result)  # 2.0
    ```

- 取模运算符：

    ```
    result = 8
    number = 4

    result %= number  # 这等同于 result = result % number

    print(result)  # 0
    ```

- 幂运算符：

    ```
    result = 2
    number = 4

    result **= number  # 这等同于 result = result ** number

    print(result)  # 16
    ```

### 比较运算符

我们在小学就学过进行数字比较，例如检查一个特定的数字是否大于另一个数字，或者它们是否相等。

我们可以在 Python 中使用几乎相同的运算符来进行此类比较。

让我们看看它们的实际应用。

#### 相等性

检查两个数字是否相等可以使用 == 来完成：

```
print(2 == 3)  # False
```

最后一个表达式求值为 False，因为 2 不等于 3。
还有另一个运算符可用于检查两个数字是否不相等。这是一个你可能在数学课上没有机会看到的运算符，其书写方式完全如此。这就是运算符 !=。
让我们比较一下 2 是否不等于 3：

```
print(2 != 3)  # True
```

这个表达式求值为 True，因为 2 确实不等于 3。

#### 不等性

现在我们将看到如何检查一个数字是否大于另一个数字：

```
print(2 > 3)  # False
```

这是你从数学课上应该已经知道的内容。
当尝试检查一个数字是否大于或等于另一个数字时，我们需要使用运算符 >=：

## 逻辑运算符

在高中数学中，你可能已经学过像 `and` 和 `or` 这样的逻辑运算符。

简而言之，当使用 `and` 时，要使表达式求值为 `True`，两个语句都必须为真。在 Python 中，我们使用 `and` 来实现它：

```python
print(5 > 0 and 3 < 5)  # True
```

这个例子将求值为 `True`，因为 5 大于 0，这求值为 `True`，并且 3 小于 5，这也求值为 `True`，由此我们得到 `True and True`，它求值为 `True`。

让我们看一个 `and` 表达式将求值为 `False` 的例子：

```python
print(2 > 5 and 0 > -1)  # False
```

2 不大于 5，所以左边的语句将求值为 `False`。无论右边是什么，整个表达式都将等于 `False`，因为 `False and whatever value else` 的结果将是 `False`。

当我们希望至少有一个语句求值为 `True` 时，我们应该使用 `or`：

```python
print(2 > 5 or 0 > -1)  # True
```

这将求值为 `True`，因为右边的语句求值为 `True`。

如果两个语句都是 `False`，那么 `or` 最终的结果是 `False`：

```python
print(2 < 0 or 0 > 1)  # False
```

这是 `False`，因为 0 不大于 2，并且 0 也不大于 1。因此，整个表达式是 `False`。

## 数据类型

### 变量

变量可以被视为你能想到的任何计算机程序的构建块。

它们可以用来存储值，然后根据需要重复使用多次。当你想改变它们的值时，你只需在一个地方更改它，你刚刚更改的新值将反映在使用此变量的所有其他地方。

Python 中的每个变量都是一个对象。

变量在用值初始化的那一刻被创建。

以下是 Python 变量的一般规则：

- 变量名必须以字母或下划线字符开头。它不能以数字开头。
- 变量名只能包含字母数字字符和下划线（A-z、0-9 和 _）。
- 变量名区分大小写，这意味着 `height`、`Height`、`HEIGHT` 都是不同的变量。

让我们定义我们的第一个变量：

```python
age = 28
```

在这个例子中，我们正在初始化一个名为 `age` 的变量，并为其赋值 28。

我们可以与其他变量一起定义，比如：

```python
age = 28
salary = 10000
```

我们可以使用几乎任何我们想要的名称，但更好的做法是使用你和与你共事的其他同事都能很好理解的名称。

Python 中还有其他变量类型，例如浮点数、字符串和布尔值。我们甚至可以创建自己的自定义类型。

让我们看一个保存浮点数的变量的例子：

```python
height = 3.5
```

如你所见，这个初始化与我们处理整数时的初始化非常相似。这里我们只是更改了右边的值，Python 解释器足够聪明，知道我们正在处理另一种类型的变量，即浮点类型的变量。

让我们看一个字符串的例子：

```python
reader = "Fatos"
```

我们将字符串值放在引号或撇号中，以指定我们想要存储在字符串变量中的值。

当我们想存储布尔值时，我们需要使用保留字，即 *True* 和 *False*。

```python
text_visibile = False
```

当我们有产生布尔值的表达式时，我们也可以存储布尔值，例如，当我们比较一个数字与另一个数字时，比如：

```python
is_greater = 5 > 6
```

这个变量将被初始化为值 `False`，因为 5 小于 6。

### 数字

Python 中有三种数字类型：整数、浮点数和复数。

#### 整数

整数表示整数，可以是正数也可以是负数，并且不包含任何小数部分。
以下是一些整数的例子：1、3000、-31234 等。
当两个整数相加、相减或相乘时，我们最终得到的结果是一个整数。

```python
print(3 + 5)  # 8
print(3 - 5)  # -2
print(3 * 5)  # 15
```

这些都是整数。
这也包括我们将一个数提升到幂的情况：

```python
result = 3 ** 4  # 这类似于将 3 * 3 * 3 * 3 相乘，在这种情况下，我们将整数相乘
print(result)  # 81
```

当我们想将两个整数相除时，我们将得到一个浮点数。

#### 布尔值

布尔类型表示真值 `True` 和 `False`。我们将这种类型的解释包含在这个 *数字* 部分中，因为布尔值确实是整数类型的子类型。

更具体地说，几乎总是 `False` 值可以被视为 0，而 `True` 值可以被视为 1。

因此，我们也可以对它们进行算术运算：

```python
print(True * 5)  # 5
print(False * 500)  # 0，因为 False 等于 0
```

布尔值的这种整数表示的例外情况是当这些值是字符串时，例如 "False" 和 "True"。

#### 浮点数

浮点数是包含小数部分的数字，例如 -3.14、12.031、9.3124 等。

我们也可以使用 `float()` 函数转换浮点数：

```python
ten = float(10)

print(ten)  # 10.0
```

当两个浮点数相加、相减或相除，或者一个浮点数和一个整数相加、相减或相除时，我们最终得到的结果是一个浮点数：

```python
print(3.4 * 2)  # 6.8
print(3.4 + 2)  # 5.4
print(3.4 - 2)  # 1.4
print(2.1 * 3.4)  # 7.14
```

#### 复数

复数具有实部和虚部，我们按以下方式书写：

```python
complex_number = 1 + 5j

print(complex_number)  # (1+5j)
```

### 字符串

字符串表示用单引号或双引号括起来的字符，两者被同等对待：

```python
name = "Fatos"  # 双引号

name = 'Fatos'  # 单引号
```

如果我们想在字符串中包含一个引号，我们需要让 Python 知道它不应该关闭字符串，而只是转义该引号：

```python
greeting = 'Hello. I\'m fine.'  # 我们转义了撇号

double_quote_greeting = "Hello. I'm fine."  # 当使用双引号时，我们不需要转义撇号
```

当我们想在字符串中包含换行符时，我们可以包含特殊字符 `\n`：

```python
my_string = "I want to continue \n in the next line"

print(my_string)
# I want to continue
# in the next line
```

由于字符串是字符数组，我们可以使用索引来索引特定字符。索引从 0 开始，一直到字符串长度减 1。我们排除等于字符串长度的索引，因为我们从 0 开始索引，而不是从 1 开始。

这是一个例子：

```python
string = "Word"
```

在这个例子中，如果我们选择字符串的各个字符，它们将如下展开：

```python
string = "Word"

print(string[0])  # W
print(string[1])  # o
print(string[2])  # r
print(string[3])  # d
```

我们也可以使用负索引，这意味着我们从字符串的末尾开始，使用 -1。我们不能使用 0 从字符串的末尾开始索引，因为 -0 = 0：

```python
print(string[-1])  # d
print(string[-2])  # r
print(string[-3])  # o
print(string[-4])  # W
```

我们还可以进行切片，只包含字符串的一部分而不是整个字符串，例如，如果我们想获取从特定索引开始到特定索引结束的字符，我们应该按以下方式编写：`string[start_index:end_index]`，不包括索引 `end_index` 处的字符：

```python
string = "Word"

print(string[0:3])  # Wor
```

如果我们想从特定索引开始并继续获取字符串的所有剩余字符直到末尾，我们可以省略指定结束索引，如下所示：

```python
string = "Word"

print(string[2:])  # rd
```

如果我们想从 0 开始直到特定索引，我们可以简单地指定结束索引：

## 字符串运算符

我们可以使用 `+` 运算符来连接字符串：

```python
first = "First"
second = "Second"

concatenated_version = first + " " + second

print(concatenated_version)  # First Second
```

我们可以将乘法运算符 `*` 与字符串和数字一起使用。这可以用来将字符串重复指定的次数。例如，如果我们想将一个字符串重复5次，但又不想手动写五遍，我们可以简单地将其与数字5相乘：

```python
string = "Abc"

repeated_version = string * 5

print(repeated_version)  # AbcAbcAbcAbcAbc
```

## 字符串内置方法

字符串有一些内置方法，我们可以使用它们来更方便地操作字符串。

### len()

`len()` 是一个我们可以用来获取字符串长度的方法：

```python
sentence = "I am fine."

print(len(sentence))  # 10
```

### replace()

`replace()` 可以用来将字符串中的一个字符或子字符串替换为另一个字符或子字符串：

```python
string = "Abc"

modified_version = string.replace("A", "Z")

print(modified_version)  # Zbc
```

### strip()

`strip()` 会移除字符串开头或结尾可能存在的空白字符：

```python
string = " Hi there "

print(string.strip())  # Hi there
```

### split()

`split()` 可以用来根据指定的分隔符模式将字符串转换为子字符串数组。例如，假设我们想将一个句子中的所有单词保存到一个单词数组中。这些单词由空格分隔，因此我们需要基于此进行分割：

```python
sentence = "This is a sentence that is being declared here"

print(sentence.split(" "))
# ['This', 'is', 'a', 'sentence', 'that', 'is', 'being', 'declared', 'here']
```

### join()

`join()` 是 `split()` 的反操作：它从一个字符串数组返回一个字符串。连接过程会在数组的每个元素之间使用指定的分隔符，最终得到一个连接后的字符串：

```python
words = ["cat", "dog", "rabbit"]

print(" - ".join(words))  # cat - dog - rabbit
```

### count()

`count()` 可以用来计算一个字符或子字符串在字符串中出现的次数：

```python
string = "Hi there"

print(string.count("h"))  # 1，因为它是区分大小写的，'h' 不等于 'H'

print(string.count("e"))  # 2

print(string.count("Hi"))  # 1

print(string.count("Hi there"))  # 1
```

### find()

`find()` 可以用来在字符串中查找一个字符或子字符串，并返回其索引。如果找不到，它将简单地返回 -1：

```python
string = "Hi there"

print(string.find("3"))  # -1

print(string.find("e"))  # 5

print(string.find("Hi"))  # 0

print(string.find("Hi there"))  # 0
```

### lower()

`lower()` 可以用来将字符串的所有字符转换为小写：

```python
string = "Hi there"

print(string.lower())  # hi there
```

### upper()

`upper()` 可以用来将字符串的所有字符转换为大写：

```python
string = "Hi there"

print(string.upper())  # HI THERE
```

### capitalize()

`capitalize()` 可以用来将字符串的第一个字符转换为大写：

```python
string = "hi there"

print(string.capitalize())  # Hi there
```

### title()

`title()` 可以用来将字符串中每个单词（由空格分隔的序列）的首字母转换为大写：

```python
string = "hi there"

print(string.title())  # Hi There
```

### isupper()

`isupper()` 是一个可以用来检查字符串中所有字符是否都是大写的方法：

```python
string = "are you HERE"
another_string = "YES"

print(string.isupper())  # False
print(another_string.isupper())  # True
```

### islower()

`islower()` 类似地可以用来检查所有字符是否都是小写：

```python
string = "are you HERE"
another_string = "no"

print(string.islower())  # False
print(another_string.islower())  # True
```

### isalpha()

如果字符串中的所有字符都是字母表中的字母，`isalpha()` 返回 True：

```python
string = "A1"
another_string = "aA"

print(string.isalpha())  # False，因为它包含 1
print(another_string.isalpha())  # True，因为 `a` 和 `A` 都是字母表中的字母
```

### isdecimal()

如果字符串中的所有字符都是数字，`isdecimal()` 返回 True：

```python
string = "A1"
another_string = "3.31"
yet_another_string = "3431"

print(string.isdecimal())  # False，因为它包含 'A'
print(another_string.isdecimal())  # False，因为它包含 '.'
print(yet_another_string.isdecimal())  # True，因为它只包含数字
```

## 格式化

格式化字符串非常有用，因为无论你正在处理什么类型的项目或脚本，你都可能经常使用它。

让我们首先说明为什么我们需要格式化，并引入字符串插值。

假设我想开发一个软件，在人们进来时向他们问好，例如：

```python
greeting = "Good morning Fatos."
```

这看起来不错，但我不只是唯一使用它的人，对吧？

我只是众多使用者中的一员。

现在，如果有人来注册，我将不得不使用他们自己的名字，例如：

```python
greeting = "Good morning Besart."
```

这只是我的第一个真实用户在注册。我还没算上我自己。

现在让我们假设我很幸运，第二个用户也在一个周五的早上出现了，我们的应用程序应该显示：

```python
greeting = "Good morning Betim."
```

正如你所看到的，从商业角度来看，我们正在取得进展，因为有两个新用户出现了，但这不是一个可扩展的实现。我们正在编写一个非常静态的问候语。

你应该已经记得我们即将提到一些在开头就介绍过的东西。

是的，我们需要使用变量，并在字符串旁边包含一个变量，如下所示：

```python
greeting = "Good morning " + first_name
```

这要灵活得多。

这是一种方法。

我提到的另一种方法是使用一个名为 `format()` 的方法。

我们可以使用花括号来指定我们想要放置动态值的位置，如下所示：

```python
greeting = "Good morning {}. Today is {}.".format("Fatos", "Friday")
```

现在，这将把 `format()` 方法的第一个参数放入第一个花括号中，在我们的例子中是 Fatos。然后，在第二个花括号出现的地方，它将放入 `format()` 方法的第二个参数。

如果我们尝试打印字符串的值，我们应该得到以下结果：

```python
print(greeting)
# Good morning Fatos. Today is Friday.
```

我们可以在花括号内使用索引来指定参数，如下所示，然后可以使用：

```python
greeting = "Today is {1}. Have a nice day {0}".format("Fatos", "Friday")

print(greeting)
# Today is Friday. Have a nice day Fatos.
```

我们也可以在 `format()` 方法内指定参数，并在花括号内使用这些特定的词作为引用：

```python
greeting = "Today is {day_of_the_week}. Have a nice day {first_name}.".format(first_name="Fatos", day_of_the_week="Friday")
print(greeting)  # Today is Friday. Have a nice day Fatos.
```

我们可以在一个示例中结合使用这两种类型的参数，如下所示：

## 列表

如果你观察一个书架，你会看到书籍被堆叠并紧密地摆放在一起。你可以看到许多收集和以某种方式组织元素的例子。这在计算机编程中也相当重要。我们不能仅仅继续声明无数的变量并轻松地管理它们。

假设我们有一个学生班级，并想保存他们的名字。我们可以开始根据他们在教室中的位置来保存他们的名字：

```
first = "Albert"
second = "Besart"
third = "Fisnik"
fourth = "Festim"
fifth = "Gazmend"
```

这个列表可以一直延续下去，而且我们也很难跟踪所有的名字。

幸运的是，我们有一种更简单的方法将这些名字放入 Python 中一个叫做列表的集合中。

让我们创建一个名为 `students` 的列表，并将上一个代码块中声明的所有名字存储其中：

```
students = ["Albert", "Besart", "Fisnik", "Festim", "Gazmend"]
```

这样更整洁，对吧？
此外，这种方式让我们更容易管理和操作这些元素。
你可能会想，“嗯，对我来说直接调用 `first` 并获取其中存储的值更容易。现在从这个名为 `students` 的新列表中获取值是不可能的。”
如果我们不能读取和使用刚刚存储在列表中的元素，那它的用处就小多了。
幸运的是，列表有索引，索引从 0 开始，这意味着如果我们想获取列表中的第一个元素，我们需要使用索引 0，而不是你可能认为的索引 1。
在上面的例子中，列表项有以下对应的索引：

```
students = ["Albert", "Besart", "Fisnik", "Festim", "Gazmend"]
# 索引          0          1          2
```

现在，如果我们想获取第一个元素，我们只需写：

```
students[0]
```

如果我们想获取第二个元素，我们只需写：

```
students[1]
```

正如你可能理解的那样，我们只需写出列表的名称，以及我们想要获取的元素的对应索引，放在方括号中即可。

这个列表当然不是静态的。我们可以向其中添加元素，比如当一个新学生加入班级时。

让我们向列表 `students` 中添加一个值为 `Besfort` 的新元素：

```
students.append("Besfort")
```

我们也可以更改现有元素的值。为此，我们只需用新值重新初始化列表的该特定元素。

例如，让我们更改第一个学生的名字：

```
students[0] = "Besim"
```

列表可以包含不同类型的变量，例如，我们可以有一个包含整数、浮点数和字符串的列表：

```
combined_list = [3.14, "An element", 1, "Another element here"]
```

## 切片

与字符串类似，列表也可以进行切片，其结果是返回新列表，这意味着原始列表保持不变。

让我们看看如何使用切片获取列表的前三个元素：

```
my_list = [1, 2, 3, 4, 5]

print(my_list[0:3])  # [1, 2, 3]
```

如你所见，我们指定了 0 作为起始索引，3 作为切片应停止的索引，不包括结束索引处的元素。

如果我们想从某个索引开始并获取列表中所有剩余的元素，即结束索引应该是最后一个索引，那么我们可以省略而不必写出最后一个索引：

```
my_list = [1, 2, 3, 4, 5]

print(my_list[3:])  # [4, 5]
```

类似地，如果我们想从列表的开头开始切片直到某个特定索引，那么我们可以完全省略写出索引 0，因为 Python 足够聪明可以推断出来：

```
my_list = [1, 2, 3, 4, 5]

print(my_list[:3])  # [1, 2, 3]
```

Python 中的字符串是不可变的，而列表是可变的，这意味着我们可以在声明列表后修改其内容。

举例说明，假设我们想更改字符串中的第一个字符，即按以下方式将 S 替换为 B：

```
string = "String"
string[0] = "B"
```

现在，如果我们尝试打印 `string`，我们会得到如下错误：

```
# TypeError: 'str' object does not support item assignment
```

现在如果我们有一个列表并想修改它的第一个元素，那么我们可以成功做到：

```
my_list = ["a", "b", "c", "d", "e"]

my_list[0] = 50

print(my_list)  # [50, 'b', 'c', 'd', 'e']
```

我们可以使用 `+` 运算符将列表与另一个列表连接来扩展列表：

```
first_list = [1, 2, 3]

second_list = [4, 5]

first_list = first_list + second_list

print(first_list)  # [1, 2, 3, 4, 5]
```

## 列表嵌套

我们可以将一个列表嵌套在另一个列表中：

```
math_points = [30, "Math"]

physics_points = [53, "Physics"]

subjects = [math_points, physics_points]

print(subjects)  # [[30, 'Math'], [53, 'Physics']]
```

这些列表甚至不需要具有相同的长度。
要访问列表中的列表元素，我们需要使用双重索引。
让我们看看如何访问 `subjects` 列表中的 `math_points` 元素。由于 `math_points` 是 `subjects` 列表中位于索引 0 的元素，我们只需执行以下操作：

```
print(subjects[0])  # [30, 'Math']
```

现在假设我们想访问 `subjects` 列表中的 Math。由于 Math 位于索引 1，我们需要使用以下双重索引：

```
print(subjects[0][1])  # 'Math'
```

## 列表方法

`len()` 是一个可用于查找列表长度的方法：

```
my_list = ["a", "b", 1, 3]

print(len(my_list))  # 4
```

## 添加元素

我们也可以通过添加新元素来扩展列表，或者我们也可以删除元素。

使用 `append()` 方法可以在列表末尾添加新元素：

```
my_list = ["a", "b", "c"]

my_list.append("New element")

my_list.append("Yet another new element")

print(my_list)
# ['a', 'b', 'c', 'New element', 'Yet another new element']
```

如果我们想在列表的特定索引处添加元素，可以使用 `insert()` 方法，其中我们在第一个参数中指定索引，在第二个参数中指定要添加到列表中的元素：

```
my_list = ["a", "b"]

my_list.insert(1, "z")

print(my_list)  # ['a', 'z', 'b']
```

## 从列表中删除元素

我们可以使用 `pop()` 方法从列表中删除元素，该方法会删除列表中的最后一个元素：

```
my_list = [1, 2, 3, 4, 5]

my_list.pop()  # 从列表中删除 5

print(my_list)  # [1, 2, 3, 4]

my_list.pop()  # 从列表中删除 4

print(my_list)  # [1, 2, 3]
```

我们也可以指定列表中某个元素的索引，以指示我们应该删除列表中的哪个元素：

```
my_list = [1, 2, 3, 4, 5]

my_list.pop(0)  # 删除索引 0 处的元素

print(my_list)  # [2, 3, 4, 5]
```

我们也可以使用 `del` 语句删除列表中的元素，然后指定要删除的元素的值：

```
my_list = [1, 2, 3, 4, 1]

del my_list[0]  # 删除元素 my_list[0]

print(my_list)  # [2, 3, 4, 5]
```

我们也可以使用 `del` 删除列表的切片：

```
my_list = [1, 2, 3, 4, 1]

del my_list[0:3]  # 删除元素：my_list[0], my_list[1], my_list[2]

print(my_list)  # [4, 1]
```

这也可以使用 `remove()` 来完成：

```
my_list = [1, 2, 3, 4]

my_list.remove(3)

print(my_list)  # [1, 2, 4]
```

`reverse()` 可用于反转列表中的元素。这非常简单直接：

## 索引搜索

使用索引获取列表中的元素很简单。查找列表中元素的索引也很容易。我们只需使用 `index()` 方法，并在列表中提及我们想要查找的元素：

```python
my_list = ["Fatos", "Morina", "Python", "Software"]

print(my_list.index("Python"))  # 2
```

## 成员关系

这非常直观，并且与现实生活相关：我们会问自己某物是否是某物的一部分。

- 我的手机是在口袋里还是在包里？
- 我同事的邮件是否包含在抄送列表中？
- 我的朋友是否在这个咖啡店里？

在 Python 中，如果我们想检查一个值是否是某物的一部分，我们可以使用 `in` 运算符：

```python
my_list = [1, 2, 3]  # 这是一个列表

print(1 in my_list)  # True
```

由于 1 包含在数组 [1, 2, 3] 中，该表达式求值为 True。
我们不仅可以将其用于数字数组，也可以用于字符数组：

```python
vowels = ['a', 'i', 'o', 'u']
print('y' in vowels)  # False
```

由于 y 不是元音字母，也不包含在声明的数组中，因此上一段代码片段第二行中的表达式将结果为 False。

类似地，我们也可以使用 `not in` 来检查某物是否不包含在内：

```python
odd_numbers = [1, 3, 5, 7]
print(2 not in odd_numbers)  # True
```

由于 2 不包含在数组中，该表达式将求值为 True。

## 排序

对列表中的元素进行排序可能是你时不时需要做的事情。`sort()` 是一个内置方法，它使你能够按字母顺序或数字顺序对列表中的元素进行升序排序：

```python
my_list = [3, 1, 2, 4, 5, 0]

my_list.sort()

print(my_list)  # [0, 1, 2, 3, 4, 5]

alphabetical_list = ['a', 'c', 'b', 'z', 'e', 'd']

alphabetical_list.sort()

print(alphabetical_list)  # ['a', 'b', 'c', 'd', 'e', 'z']
```

我们在此未包含列表的其他方法。

## 列表推导式

列表推导式代表了一种简洁的方式，我们使用 `for` 循环从现有列表创建一个新列表。最终结果总是一个新列表。

让我们从一个例子开始，我们想将列表中的每个数字乘以 10，并将结果保存在一个新列表中。首先，让我们在不使用列表推导式的情况下完成此操作：

```python
numbers = [2, 4, 6, 8]  # 完整列表

numbers_tenfold = []  # 空列表

for number in numbers:
    number = number * 10  # 将每个数字乘以 10
    numbers_tenfold.append(number)  # 将该新数字添加到新列表中

print(numbers_tenfold)  # [20, 40, 60, 80]
```

我们可以使用列表推导式以以下方式实现：

```python
numbers = [2, 4, 6, 8]  # 完整列表

numbers_tenfold = [number * 10 for number in numbers]  # 列表推导式

print(numbers_tenfold)  # [20, 40, 60, 80]
```

在进行这些列表推导式时，我们也可以包含条件。

假设我们想保存一个正数列表。

在我们编写使用列表推导式实现此功能的方法之前，让我们先编写一种方法，从另一个列表中创建一个仅包含大于 0 的数字的列表，并将这些正数增加 100：

```python
positive_numbers = []  # 空列表

numbers = [-1, 0, 1, -2, -3, -4, 3, 2]  # 完整列表

for number in numbers:
    if number > 0:  # 如果当前数字大于 0
        positive_numbers.append(number + 100)  # 将该数字添加到列表 positive_numbers 中

print(positive_numbers)  # [101, 103, 102]
```

我们可以使用列表推导式完成相同的操作：

```python
numbers = [-1, 0, 1, -2, -3, -4, 3, 2]  # 完整列表

positive_numbers = [number + 100 for number in numbers if number > 0]  # 列表推导式

print(positive_numbers)  # [101, 103, 102]
```

如你所见，这要短得多，并且编写起来应该花费更少的时间。

我们也可以将列表推导式与多个列表一起使用。

让我们举一个例子，我们想将一个列表中的每个元素与另一个列表中的每个元素相加：

```python
first_list = [1, 2, 3]
second_list = [50]

double_lists = [first_element +
                second_element for first_element in first_list for second_element in second_list]

print(double_lists)  # [51, 52, 53]
```

最终，我们将得到一个结果列表，其元素数量与最长列表的元素数量相同。

## 元组

元组是有序且不可变的集合，这意味着它们的内容无法更改。它们是有序的，其元素可以使用索引访问。

让我们从我们的第一个元组开始：

```python
vehicles = ("Computer", "Smartphone", "Smart watch", "Tablet")

print(vehicles)

# ('Computer', 'Smartphone', 'Smart watch', 'Tablet')
```

我们在列表章节中看到的所有索引和切片操作也适用于元组：

```python
print(len(vehicles))  # 4

print(vehicles[3])  # Tablet

print(vehicles[:3])  # ('Computer', 'Smartphone', 'Smart watch')
```

查找元组中元素的索引可以使用 `index()` 方法完成：

```python
print(vehicles.index('tablet'))  # 3
```

我们也可以使用 `+` 运算符连接或合并两个元组：

```python
natural_sciences = ('Chemistry', 'Astronomy',
                    'Earth science', 'Physics', 'Biology')

social_sciences = ('Anthropology', 'Archaeology', 'Economics', 'Geography',
                   'History', 'Law', 'Linguistics', 'Politics', 'Psychology', 'Sociology')

sciences = natural_sciences + social_sciences

print(sciences)
# ('Chemistry', 'Astronomy', 'Earth science', 'Physics', 'Biology', 'Anthropology', 'Archaeology', 'Economics', 'Geography', 'History', 'Law', 'Linguistics', 'Politics', 'Psychology', 'Sociology')
```

## 成员检查

我们可以使用 `in` 和 `not in` 运算符检查元素是否是元组的一部分，就像列表一样：

```python
vehicles = ('Car', 'Bike', 'Airplane')

print('Motorcycle' in vehicles)  # False，因为 Motorcycle 不包含在 vehicles 中

print('Train' not in vehicles)  # True，因为 Train 不包含在 vehicles 中
```

## 嵌套 2 个元组

除了合并，我们还可以通过将要嵌套的元组放在括号内，将元组嵌套到单个元组中：

```python
natural_sciences = ('Chemistry', 'Astronomy',
                    'Earth science', 'Physics', 'Biology')

social_sciences = ('Anthropology', 'Archaeology', 'Economics', 'Geography',
                  'History', 'Law', 'Linguistics', 'Politics', 'Psychology', 'Sociology')

sciences = (natural_sciences, social_sciences)

print(sciences)
# (('Chemistry', 'Astronomy', 'Earth science', 'Physics', 'Biology'), ('Anthropology', 'Archaeology', 'Economics', 'Geography', 'History', 'Law', 'Linguistics', 'Politics', 'Psychology', 'Sociology'))
```

## 不可变性

由于元组是不可变的，因此在创建后无法更改它们。这意味着我们不能在其中添加或删除元素，也不能将一个元组附加到另一个元组。

我们甚至不能修改其中现有的元素。如果我们尝试修改元组中的元素，我们将面临如下问题：

```python
vehicles = ('Car', 'Bike', 'Airplane')

vehicles[0] = 'Truck'

print(vehicles)
# TypeError: 'tuple' object does not support item assignment
```

## 字典：键值数据结构

正如我们之前看到的，列表中的元素与可用于引用这些元素的索引相关联。Python 中还有另一种数据结构，允许我们指定自己的自定义索引，而不仅仅是数字。这些被称为字典，它们类似于我们用来查找不理解的外语单词含义的字典。

假设你正在尝试学习德语，并且有一个你之前没有机会学习的新单词，你刚刚在市场上看到：Wasser。

现在，你可以拿起手机，使用 Google Translate 或你选择的任何其他应用程序检查其对应的英文含义，但如果你使用的是实体字典，你需要找到这个词，翻到特定页面，并在旁边检查其含义。这个词含义的参考或关键就是术语 Wasser。

现在，如果我们想在 Python 中实现这一点，我们不应该使用索引仅为数字的列表。我们应该使用字典。

相反。

对于字典，我们使用花括号，每个元素包含两部分：键和值。在我们之前的例子中，键是德语单词，而值是其对应的英语翻译，如下例所示：

```
german_to_english_dictionary = {
    "Wasser": "Water",
    "Brot": "Bread",
    "Milch": "Milk"
}
```

现在，当我们想要访问字典中的特定元素时，我们只需使用键。例如，假设我们想获取单词 Brot 的英语含义。为此，我们可以简单地使用该键来引用该元素：

```
brot_translation = german_to_english_dictionary["Brot"]
print(translation)  # Bread
```

当我们打印获取到的值时，我们将得到英语翻译。

类似地，我们可以通过获取以 Milch 为键的元素的值来获取单词 Milch 的英语翻译：

```
milch_translation = german_to_english_dictionary["Milch"]
print(milch_translation)  # Milk
```

我们也可以使用 `get()` 方法并指定要获取的项目的键来获取字典中元素的值：

```
german_to_english_dictionary.get("Wasser")
```

我们可以使用 `dict()` 创建字典：

```
python
words = dict([
    ('abandon', 'to give up to someone or something on the ground'),
    ('abase', 'to lower in rank, office, or esteem'),
    ('abash', 'to destroy the self-possession or self-confidence of')
])

print(words)
# {'abandon': 'to give up to someone or something on the ground', 'abase': 'to lower in rank, office, or esteem', 'abash': 'to destroy the self-possession or self-confidence of'}
```

键和值可以是任何数据类型。
我们可以有多个键对应相同的值，但键必须是唯一的。

## 添加新值

我们可以通过指定一个新的键和相应的值来在字典中添加新值，然后 Python 将在该字典中创建一个新元素：

```
python
words = {
    'a': 'alfa',
    'b': 'beta',
    'd': 'delta',
}

words['g'] = 'gama'

print(words)
# {'a': 'alfa', 'b': 'beta', 'd': 'delta', 'g': 'gama'}
```

如果我们指定一个已经是字典一部分的元素的键，该元素将被修改：

```
1 words = {
2     'a': 'alfa',
3     'b': 'beta',
4     'd': 'delta',
5 }
6
7 words['b'] = 'bravo'
8
9
10 print(words)
11 # {'a': 'alfa', 'b': 'bravo', 'd': 'delta'}
```

## 删除元素

如果我们想从字典中删除元素，可以使用 `pop()` 方法，并指定要删除的元素的键：

```
1 words = {
2     'a': 'alfa',
3     'b': 'beta',
4     'd': 'delta',
5 }
6
7 words.pop('a')
8
9 print(words)  # {'b': 'beta', 'd': 'delta'}
```

我们也可以使用 `popitem()` 删除值，从 Python 3.7 开始，它会删除最后插入的键值对。在早期版本中，它会删除一个随机的对：

```
1 words = {
2     'a': 'alfa',
3     'b': 'beta',
4     'd': 'delta',
5 }
6
7 words['g'] = 'gamma'
8
9 words.popitem()
10
11 print(words)
12 # {'a': 'alfa', 'b': 'beta', 'd': 'delta'}
```

还有另一种删除元素的方法，即使用 `del` 语句：

```
1 words = {
2     'a': 'alfa',
3     'b': 'beta',
4     'd': 'delta',
5 }
6
7 del words['b']
8
9 print(words)  # {'a': 'alfa', 'd': 'delta'}
```

## 长度

我们可以像处理列表和元组一样，使用 `len()` 获取字典的长度：

```
1 words = {
2     'a': 'alfa',
3     'b': 'beta',
4     'd': 'delta',
5 }
6
7 print(len(words))  # 3
```

## 成员关系

如果我们想检查一个键是否已经是字典的一部分，以避免覆盖它，我们可以像处理列表和元组一样使用 `in` 和 `not in` 运算符：

```
1 words = {
2     'a': 'alfa',
3     'b': 'beta',
4     'd': 'delta',
5 }
6
7 print('a' in words)  # True
8 print('z' not in words)  # True
```

## 推导式

我们可以像处理列表一样使用推导式来快速创建字典。

为此，我们需要使用一个名为 `items()` 的方法，该方法将字典转换为元组列表，其中索引 0 处的元素是键，索引 1 处的元素是值。

让我们先看看 `items()` 方法的实际应用：

```
points = {
    'Festim': 50,
    'Zgjim': 89,
    'Durim': 73
}

elements = points.items()

print(elements) # dict_items([('Festim', 50), ('Zgjim', 89), ('Durim', 73)])
```

现在让我们使用推导式从现有的 `points` 字典创建一个新字典。我们可以假设一位教授心情很好，慷慨地奖励每位学生 10 分。我们想通过将这些新分数保存在一个新字典中来为每位学生添加这些新分数：

```
points = {
    'Festim': 50,
    'Zgjim': 89,
    'Durim': 73
}

elements = points.items()

points_modified = {key: value + 10 for (key, value) in elements}

print(points_modified) # {'Festim': 60, 'Zgjim': 99, 'Durim': 83}
```

# 集合

集合是无序且无索引的数据集合。由于集合中的元素是无序的，我们不能使用索引或 `get()` 方法来访问元素。

我们可以添加元组，但不能在集合中添加字典或列表。

我们不能在集合中添加重复的元素。这意味着当我们想从另一种类型的集合中删除重复元素时，我们可以利用集合的这种唯一性。

让我们开始使用花括号创建我们的第一个集合：

```
first_set = {1, 2, 3}
```

我们也可以使用 `set()` 构造函数创建集合：

```
empty_set = set()  # 空集合

first_set = set((1, 2, 3))  # 我们正在将元组转换为集合
```

像所有数据结构一样，我们可以使用 `len()` 方法找到集合的长度：

```
print(len(first_set))  # 3
```

## 添加元素

向集合中添加一个元素可以使用 `add()` 方法完成：

```
1 my_set = {1, 2, 3}
2
3 my_set.add(4)
4
5 print(my_set)  # {1, 2, 3, 4}
```

如果我们想添加多个元素，那么我们需要使用 `update()` 方法，并将列表、元组、字符串或另一个集合作为该方法的输入：

```
1 my_set = {1, 2, 3}
2
3 my_set.update([4, 5, 6])
4
5 print(my_set)  # {1, 2, 3, 4, 5, 6}
6
7 my_set.update("ABC")
8
9 print(my_set)  # {1, 2, 3, 4, 5, 6, 'A', 'C', 'B'}
```

## 删除元素

如果我们想从集合中删除元素，可以使用 `discard()` 或 `remove()` 方法：

```
1 my_set = {1, 2, 3}
2
3 my_set.remove(2)
4
5 print(my_set)  # {1, 3}
```

如果我们尝试使用 `remove()` 删除一个不是集合一部分的元素，那么我们将得到一个错误：

```
1 my_set = {1, 2, 3}
2
3 my_set.remove(4)
4
5 print(my_set)  # KeyError: 4
```

为了避免在从集合中删除元素时出现此类错误，我们可以使用 `discard()` 方法：

```
1 my_set = {1, 2, 3}
2
3 my_set.discard(4)
4
5 print(my_set)  # {1, 2, 3}
```

## 集合论运算

如果你还记得高中数学课，你应该已经知道并集、交集以及两个元素集合之间的差集。Python 中的集合也支持这些运算。

### 并集

并集表示两个集合中所有唯一元素的集合。我们可以使用管道运算符 `|` 或 `union()` 方法找到两个集合的并集：

```
1 first_set = {1, 2}
2 second_set = {2, 3, 4}
3
4 union_set = first_set.union(second_set)
5
6 print(union_set)  # {1, 2, 3, 4}
```

### 交集

交集表示包含同时存在于两个集合中的元素的集合。我们可以使用 `&` 运算符或 `intersection()` 方法找到它：

```
1  first_set = {1, 2}
2  second_set = {2, 3, 4}
3
4  intersection_set = first_set.intersection(second_set)
5
6  print(union_set)  # {2}
```

### 差集

两个集合之间的差集表示仅包含存在于第一个集合中但不在第二个集合中的元素的集合。我们可以使用 `-` 运算符或 `difference()` 方法找到两个集合的差集：

```
1  first_set = {1, 2}
2  second_set = {2, 3, 4}
3
4  difference_set = first_set.difference(second_set)
5
6  print(difference_set)  # {1}
```

正如你可能从高中记得的那样，当我们求两个集合的差集时，集合的顺序很重要，而并集和交集则不是这样。

这类似于算术，其中 3 - 4 不等于 4 - 3：

## 类型转换

### 基本类型之间的转换

Python 是一门面向对象的编程语言。这就是为什么它使用类的构造函数来完成从一种类型到另一种类型的转换。

### int()

`int()` 是一个用于将整数字面量、浮点数字面量（将其四舍五入到前一个整数，即 3.1 变为 3）或字符串字面量（前提是字符串表示一个整数或浮点数字面量）进行转换的方法：

```python
three = int(3)  # 将整数字面量转换为整数
print(three)  # 3

four = int(4.8)  # 将浮点数转换为其前一个最接近的整数
print(four)  # 4

five = int('5')  # 将字符串转换为整数
print(five)  # 5
```

### float()

`float()` 类似地用于从整数、浮点数或字符串字面量创建浮点数（前提是字符串表示一个整数或浮点数字面量）：

```python
int_literal = float(5)
print(int_literal)  # 5.0

float_literal = float(1.618)
print(float_literal)  # 1.618

string_int = float("40")
print(string_int)  # 40.0

string_float = float("37.2")
print(string_float)  # 37.2
```

### str()

`str()` 可用于从字符串、整数字面量、浮点数字面量以及许多其他数据类型创建字符串：

```python
int_to_string = str(3)
print(int_to_string)  # '3'

float_to_string = str(3.14)
print(float_to_string)  # '3.14'

string_to_string = str('hello')
print(string_to_string)  # 'hello'
```

### 其他转换

我们可以将一种数据结构类型转换为另一种类型的方式如下：

```python
目标类型(输入类型)
```

让我们从具体类型开始，这样会清晰得多。

### 转换为列表

我们可以使用 `list()` 构造函数将集合、元组或字典转换为列表。

```python
books_tuple = ('Book 1', 'Book 2', 'Book 3')
tuple_to_list = list(books_tuple)  # 将元组转换为列表
print(tuple_to_list)  # ['Book 1', 'Book 2', 'Book 3']

books_set = {'Book 1', 'Book 2', 'Book 3'}
set_to_list = list(books_set)  # 将集合转换为列表
print(set_to_list)  # ['Book 1', 'Book 2', 'Book 3']
```

将字典转换为列表时，只有其键会进入列表：

```python
books_dict = {'1': 'Book 1', '2': 'Book 2', '3': 'Book 3'}
dict_to_list = list(books_dict)  # 将字典转换为列表
print(dict_to_list)  # ['1', '2', '3']
```

如果我们想保留字典的键和值，我们需要使用 `items()` 方法首先将其转换为元组列表，其中每个元组是一个键和一个值：

```python
books_dict = {'1': 'Book 1', '2': 'Book 2', '3': 'Book 3'}
dict_to_list = list(books_dict.items())  # 将字典转换为列表
print(dict_to_list)
# [('1', 'Book 1'), ('2', 'Book 2'), ('3', 'Book 3')]
```

### 转换为元组

所有数据结构都可以使用 `tuple()` 构造函数方法转换为元组，包括字典，这种情况下我们得到一个包含字典键的元组：

```python
books_list = ['Book 1', 'Book 2', 'Book 3']
list_to_tuple = tuple(books_list)  # 将列表转换为元组
print(list_to_tuple)  # ('Book 1', 'Book 2', 'Book 3')

books_set = {'Book 1', 'Book 2', 'Book 3'}
set_to_tuple = tuple(books_set)  # 将集合转换为元组
print(set_to_tuple)  # ('Book 1', 'Book 2', 'Book 3')

books_dict = {'1': 'Book 1', '2': 'Book 2', '3': 'Book 3'}
dict_to_tuple = tuple(books_dict)  # 将字典转换为元组
print(dict_to_tuple)  # ('1', '2', '3')
```

### 转换为集合

类似地，所有数据结构都可以使用 `set()` 构造函数方法转换为集合，包括字典，这种情况下我们得到一个包含字典键的集合：

```python
books_list = ['Book 1', 'Book 2', 'Book 3']
list_to_set = set(books_list)  # 将列表转换为集合
print(list_to_set)  # {'Book 2', 'Book 3', 'Book 1'}

books_tuple = ('Book 1', 'Book 2', 'Book 3')
tuple_to_set = set(books_tuple)  # 将元组转换为集合
print(tuple_to_set)  # {'Book 2', 'Book 3', 'Book 1'}

books_dict = {'1': 'Book 1', '2': 'Book 2', '3': 'Book 3'}
dict_to_set = set(books_dict)  # 将字典转换为集合
print(dict_to_set)  # {'1', '3', '2'}
```

### 转换为字典

转换为字典不能使用任何类型的集合、列表或元组，因为字典表示的数据结构中每个元素都包含一个键和一个值。

如果列表中的每个元素也是一个包含两个元素的列表，或者一个包含两个元素的元组，那么可以将列表或元组转换为字典。

```python
books_tuple_list = [(1, 'Book 1'), (2, 'Book 2'), (3, 'Book 3')]
tuple_list_to_dictionary = dict(books_tuple_list)  # 将列表转换为字典
print(tuple_list_to_dictionary)  # {1: 'Book 1', 2: 'Book 2', 3: 'Book 3'}

books_list_list = [[1, 'Book 1'], [2, 'Book 2'], [3, 'Book 3']]
tuple_list_to_dictionary = dict(books_list_list)  # 将列表转换为字典
print(tuple_list_to_dictionary)  # {1: 'Book 1', 2: 'Book 2', 3: 'Book 3'}
```

```python
books_tuple_list = ([1, 'Book 1'], [2, 'Book 2'], [3, 'Book 3'])
tuple_list_to_set = dict(books_tuple_list)  # 将元组转换为字典
print(tuple_list_to_set)  # {'Book 2', 'Book 3', 'Book 1'}
```

```python
books_list_list = ([1, 'Book 1'], [2, 'Book 2'], [3, 'Book 3'])
list_list_to_set = dict(books_list_list)  # 将元组转换为字典
print(list_list_to_set)  # {'Book 2', 'Book 3', 'Book 1'}
```

如果我们想将集合转换为字典，我们需要每个元素都是一个长度为 2 的元组。

```python
books_tuple_set = {('1', 'Book 1'), ('2', 'Book 2'), ('3', 'Book 3')}
tuple_set_to_dict = dict(books_tuple_set)  # 将集合转换为字典
print(tuple_set_to_dict)  # {'1': 'Book 1', '2': 'Book 2', '3': 'Book 3'}
```

如果我们尝试将一个每个元素都是长度为 2 的列表的集合转换为字典，我们将得到一个错误：

```python
books_list_set = {['1', 'Book 1'], ['2', 'Book 2'], ['3', 'Book 3']}
list_set_to_dict = dict(books_list_set)  # 将集合转换为字典
print(list_set_to_dict)  # {'1': 'Book 1', '2': 'Book 2', '3': 'Book 3'}
```

运行最后一个代码块后，我们将得到一个错误：

```
TypeError: unhashable type: 'list'
```

## 总结

总之，Python 有多种数据类型可用于存储数据。了解这些数据类型很重要，这样你才能根据需要选择正确的类型。请确保为手头的任务使用正确的数据类型，以避免错误并优化性能。

## 控制流

### 条件语句

当你思考我们思考和彼此交流的方式时，你可能会觉得我们确实总是在使用条件。

- 如果是早上 8 点，我就乘公交车去上班。
- 如果我饿了，我就吃东西。
- 如果这件商品便宜，我就买得起。

这在编程中也是可以做到的。我们可以使用条件来控制执行流程。
为此，我们使用保留字 `if` 和一个计算结果为 True 或 False 的表达式。我们还可以使用 `else` 语句，当 `if` 条件不满足时，我们希望流程继续执行。
为了更容易理解，让我们假设有一个例子，我们想检查一个数字是否为正数：

```python
if number > 0:
    print("The given number is positive")
else:
    print("The given number is not positive")
```

如果 `number = 2`：我们将进入 `if` 分支并执行用于在控制台打印以下文本的命令：

The given number is positive

如果我们有另一个数字，比如 -1，我们将在控制台看到以下消息被打印：

The given number **is not** positive

我们还可以使用 `elif` 添加额外的条件，而不仅仅是上面的 2 个，`elif` 在 `if` 表达式未被计算时进行计算。

让我们看一个例子，让你更容易理解：

```python
if number > 0:
    print("The given number is positive")
elif number == 0:
    print("The given number is 0")
else:
    print("The given number is negative")
```

现在，如果 `number = 0`，第一个条件将不会满足，因为该值不大于 0。你可以猜到，由于给定的数字等于 0，我们将在控制台看到以下消息被打印：

The given number is 0

在值为负数的情况下，我们的程序将通过前两个条件，因为它们不满足，然后跳转到 `else` 分支并在控制台打印以下消息：

The given number is negative

## 循环 / 迭代器

循环代表程序反复执行一组指令的能力，直到满足特定条件。我们可以使用 `while` 和 `for` 来实现。

让我们先看看使用 `for` 的迭代。

### for 循环

这种循环方式简单且非常直观。你只需指定一个起始状态，并提及它应该迭代的范围，如下例所示：

```python
for number in range(1, 7):
    print(number)
```

在这个例子中，我们从 1 迭代到 7，并在控制台中打印每个数字（从 1 到 7，不包括 7）。

我们可以根据需要更改范围中的起始和结束数字。这样，我们就可以根据具体场景灵活调整。

### while 循环

现在让我们描述使用 `while` 的迭代。这是另一种进行迭代的方式，同样非常直观。这里我们需要在 while 块之前指定一个起始条件，并相应地更新条件。`while` 循环需要一个“循环条件”。如果条件保持为 True，它就会继续迭代。在这个例子中，当 `num` 为 11 时，`循环条件` 等于 `False`。

```python
number = 1

while number < 7:
    print(number)
    number += 1  # 这部分是必要的，以防止迭代永远持续下去
```

这个 while 块将打印与我们使用 for 块时相同的语句。

## 迭代：遍历数据结构

既然我们已经介绍了迭代和列表，我们可以开始探讨遍历列表的方法。

我们不仅仅将数据存储到数据结构中然后就放任不管。我们应该能够在不同的场景中使用这些元素。

让我们使用之前的学生列表：

```python
students = ["Albert", "Besart", "Fisnik", "Festim", "Gazmend"]
```

现在，要遍历这个列表，我们只需输入：

```python
for student in students:
    print(student)
```

是的，就这么简单。我们遍历列表中的每个元素并打印它们的值。

我们也可以对字典执行此操作。然而，由于字典中的元素有两部分（键和值），我们需要同时指定键和值，如下所示：

```python
german_to_english_dictionary = {
    "Wasser": "Water",
    "Brot": "Bread",
    "Milch": "Milk"
}

for key, value in german_to_english_dictionary:
    print("The German word " + key + " means " + value + " in English")
```

我们也可以只获取字典元素的键：

```python
for key in german_to_english_dictionary:
    print(key)
```

请注意，`key` 和 `value` 只是我们选择用来说明迭代的变量名，我们可以为变量使用任何我们想要的名字，例如以下示例：

```python
for german_word, english_translation in german_to_english_dictionary:
    print("The German word " + german_word + " means " + english_translation + " in English")
```

这次迭代将在控制台中打印与上一个代码块相同的内容。

我们还可以使用嵌套的 for 循环，例如，假设我们想要遍历一个数字列表，并找出每个元素与列表中其他每个元素的和。我们可以使用嵌套的 for 循环来实现：

```python
numbers = [1, 2, 3]
sum_of_numbers = []  # 空列表

for first_number in numbers:
    for second_number in numbers:  # 遍历列表并相加数字
        current_sum = first_number + second_number
        # 将第一个列表中的当前 first_number 添加到第二个列表中的 second_number
        sum_of_numbers.append(current_sum)

print(sum_of_numbers)
# [2, 3, 4, 3, 4, 5, 4, 5, 6]
```

## 停止 for 循环

有时我们可能需要在 for 循环到达末尾之前退出。当满足某个条件，或者我们已经找到了正在寻找的东西，没有必要继续下去时，就可能出现这种情况。

在这些情况下，我们可以使用 `break` 来停止 for 循环的任何后续迭代。

假设我们想要检查列表中是否存在负数。如果我们找到了正数，我们就停止搜索。

让我们使用 `break` 来实现这一点：

```python
my_list = [1, 2, -3, 4, 0]

for element in my_list:
    print("Current number: ", element)
    if element < 0:
        print("We just found a negative number")
        break

# Current number:  1
# Current number:  2
# Current number:  -3
# We just found a negative number
```

正如我们所看到的，当我们到达 -3 时，我们从 for 循环中跳出并停止。

## 跳过迭代

也可能出现我们想要跳过某些迭代的情况，因为我们对它们不感兴趣，或者它们不那么重要。我们可以使用 `continue` 来实现这一点，它会阻止执行该代码块中其下方的代码，并将执行过程转向下一次迭代：

```python
my_sum = 0
my_list = [1, 2, -3, 4, 0]

for element in my_list:
    if element < 0:  # 不将负数包含在总和中
        continue
    my_sum += element

print(my_sum)  # 7
```

`pass` 是一个语句，可以在我们即将实现一个方法或某些东西但尚未完成且不想出错时使用。

它帮助我们即使在代码某些部分缺失的情况下也能执行程序：

```python
my_list = [1, 2, 3]

for element in my_list:
    pass  # 什么都不做
```

## 总结

总之，Python 提供了条件语句来帮助你控制程序的流程。`if` 语句让你仅在满足特定条件时运行一段代码。`elif` 语句让你仅在满足另一个条件时运行一段代码。而 `else` 语句让你仅在没有其他条件满足时运行一段代码。这些语句对于控制程序的流程非常有用。

## 函数

在很多情况下，我们需要反复使用相同的代码块。我们的第一反应可能是想写多少次就写多少次。

客观地说，这确实可行，但事实是，这是一种非常糟糕的做法。我们在做重复性的工作，这可能相当枯燥，而且容易犯很多可能被忽视的错误。

这就是为什么我们需要开始使用可以定义一次并在其他任何地方使用相同指令代码的代码块。

想想现实生活中的例子：你看到一个 YouTube 视频，它被录制并上传到 YouTube 一次。然后它会被许多其他人观看，但视频仍然是最初上传的那个。

换句话说，我们使用方法作为一组编码指令的代表，这些指令随后应该在代码中的任何其他地方被调用，而我们不必重复编写它。当我们想要修改这个方法时，我们只需在它首次声明的地方进行更改，而调用它的其他地方则无需做任何事情。

要在 Python 中定义一个方法，我们首先使用 `def` 关键字，然后是函数名，接着是我们期望使用的参数列表。之后，我们需要在缩进后的新行中开始编写方法体。

```python
def add(first_number, second_number):
    our_sum = first_number + second_number
    return our_sum
```

从颜色可以看出，`def` 和 `return` 都是 Python 中的关键字，你不能用它们来命名你的变量。

现在，无论我们想在哪里调用这个 `add()`，我们都可以在那里调用它，而不必担心完全实现它。

既然我们已经定义了这个方法，我们可以这样调用它：

```python
result = add(1, 5)

print(result)  # 6
```

你可能会觉得这是一个如此简单的方法，并开始问，我们为什么要费心为它写一个方法？
你说得对。这是一个非常简单的方法，只是为了向你介绍我们可以实现函数的方式。
让我们写一个函数，找出两个指定数字之间所有数字的和：

```python
def sum_in_range(starting_number, ending_number):
    result = 0

    while starting_number < ending_number:
        result = result + starting_number
        starting_number = starting_number + 1

    return result
```

现在，这是一组你可以在其他地方调用的指令，而无需再次编写所有内容。

## 默认参数

当我们调用函数时，可以通过在函数头部为参数设置初始值，使某些参数变为可选参数。

让我们以获取用户名字作为必需参数，第二个参数作为可选参数为例。

```python
def get_user(first_name, last_name=""):
    return f"Hi {first_name} {last_name}"
```

现在，我们使用两个参数来调用此函数：

```python
user = get_user("Durim", "Gashi")
print(user)  # Hi Durim Gashi
```

即使不指定第二个参数，我们也可以调用同一个函数：

```python
user = get_user("Durim")
print(user)  # Hi Durim
```

## 关键字参数列表

我们可以将函数的参数定义为关键字：

```python
# 第一个参数是必需的。其他两个是可选的
def get_user(number, first_name='', last_name=''):
    return f"Hi {first_name} {last_name}"
```

现在，我们可以通过将参数写为关键字来调用此函数：

```python
user = get_user(1, last_name="Gashi")
print(user)  # Hi Gashi
```

如你所见，我们可以省略 `first_name`，因为它不是必需的。在调用函数时，我们还可以改变参数的顺序，它仍然会正常工作：

```python
user = get_user(1, last_name="Gashi", first_name='Durim')
print(user)  # Hi Durim Gashi
```

## 数据生命周期

在函数内部声明的变量无法在函数外部访问。它们是隔离的。

让我们看一个例子来说明这一点：

```python
def counting():
    count = 0  # 这在函数外部无法访问。

counting()
print(count)  # 执行时会抛出错误，因为 count 仅在函数内部声明，在函数外部无法访问
```

同样，我们无法更改在函数外部声明且未作为参数传递的函数内部变量：

```python
count = 3331

def counting():
    count = 0  # 这是一个新变量

counting()
print(count)  # 3331
# 这是在函数外部声明的，没有被更改
```

## 在函数内部更改数据

我们可以更改通过函数参数传递的可变数据。可变数据代表即使在声明后也可以修改的数据，例如列表就是可变数据。

```python
names = ["betim", "durim", "gezim"]

def capitalize_names(current_list):
    for i in range(len(current_list)):
        current_list[i] = current_list[i].capitalize()
    print("Inside the function:", current_list)
    return current_list

capitalize_names(names)  # Inside the function: ['Betim', 'Durim', 'Gezim']
print("Outside the function:", names)  # Outside the function: ['Betim', 'Durim', 'Gezim']
```

对于不可变数据，我们只能在函数内部修改变量，但函数外部的实际值将保持不变。不可变数据是字符串和数字：

```python
name = "Betim"

def say_hello(current_param):
    current_param = current_param + " Gashi"
    name = current_param  # name 是一个局部变量
    print("Value inside the function:", name)
    return current_param

say_hello(name)  # Value inside the function: Betim Gashi
print("Value outside the function:", name)  # Value outside the function: Betim
```

如果我们确实想通过函数更新不可变变量，可以将函数的返回值赋给该不可变变量：

```python
name = "Betim"

def say_hello(current_param):
    current_param = current_param + " Gashi"
    name = current_param  # name 是一个局部变量
    print("Value inside the function", name)
    return current_param

# 这里我们将 name 的值赋给从函数返回的 current_param
name = say_hello(name)  # Value inside the function Betim Gashi
# Value outside the function: Betim Gashi
print("Value outside the function:", name)
```

## Lambda 函数

Lambda 函数是匿名函数，可用于返回输出。我们可以使用以下语法模式编写 lambda 函数：

```python
lambda parameters: expression
```

表达式只能写在一行中。
让我们用几个例子来开始说明这些匿名函数。
首先，我们从一个将每个输入乘以 10 的函数开始：

```python
tenfold = lambda number : number * 10
print(tenfold(10))  # 100
```

让我们再写一个例子，检查给定的参数是否为正数：

```python
is_positive = lambda a : f'{a} is positive' if a > 0 else f'{a} is not positive'
print(is_positive(3))  # 3 is positive
print(is_positive(-1))  # -1 is not positive
```

请注意，在 lambda 函数内部，我们不能在没有 else 子句的情况下使用 if 子句。
此时，你可能会想，既然 lambda 函数看起来与其他函数几乎相同，为什么我们需要使用它们。
我们可以在下一节中看到说明。

## 函数作为函数的参数

到目前为止，我们已经看到了使用数字和字符串调用函数的方法。实际上，我们可以使用任何类型的 Python 对象来调用函数。
我们甚至可以将整个函数作为函数的参数提供，这可以提供一种非常有用的抽象级别。
让我们看一个例子，我们想将一种单位转换为另一种单位：

```python
def convert_to_meters(feet):
    return feet * 0.3048

def convert_to_feet(meters):
    return meters / 0.3048

def convert_to_miles(kilometers):
    return kilometers / 1.609344

def convert_to_kilometers(miles):
    return miles * 1.609344
```

现在，我们可以创建一个通用函数，并将另一个函数作为参数传递：

```python
def conversion(operation, argument):
    return operation(argument)
```

现在我们可以像下面这样调用 `conversion()`：

```python
result = conversion(convert_to_miles, 10)
print(result)  # 6.2137119223733395
```

如你所见，我们将 `convert_to_miles` 作为函数 `conversion()` 的参数编写。我们可以像这样使用其他已定义的函数：

```python
result = conversion(convert_to_feet, 310)
print(result)  # 1017.0603674540682
```

现在我们可以利用 lambda 函数，使这种类型的抽象变得更加简单。
我们不必编写所有这四个函数，只需编写一个简洁的 lambda 函数，并在调用 `conversion()` 函数时将其用作参数：

```python
def conversion(operation, argument):
    return operation(argument)

result = conversion(lambda kilometers: kilometers / 1.609344, 10)
print(result)  # 6.2137119223733395
```

这当然更简单。
让我们使用一些其他内置函数的例子。

## map()

`map()` 是一个内置函数，它通过对现有列表的每个元素调用一个函数来获取结果，从而创建一个新对象：

```python
map(function_name, my_list)
```

让我们看一个将 lambda 函数作为 map 的函数的例子。
让我们使用列表推导将列表中的每个数字乘以三：

```python
my_list = [1, 2, 3, 4]
triple_list = [x * 3 for x in my_list]
print(triple_list)  # [3, 6, 9, 12]
```

我们可以使用 `map()` 函数和 lambda 函数来实现这一点：

```python
my_list = [1, 2, 3, 4]
triple_list = map(lambda x: x * 3, my_list)
print(triple_list)  # [3, 6, 9, 12]
```

这会创建一个新列表。旧列表没有被更改。

## filter()

这是另一个内置函数，可用于过滤满足条件的列表元素。
让我们首先使用列表推导从列表中过滤掉负元素：

```python
my_list = [3, -1, 2, 0, 14]
non_negative_list = [x for x in my_list if x >= 0]
print(non_negative_list)  # [3, 2, 0, 14]
```

现在，让我们使用 `filter()` 和 lambda 函数来过滤元素。此函数返回一个对象，我们可以使用 `list()` 将其转换为列表：

```
my_list = [3, -1, 2, 0, 14]

non_negative_filter_object = filter(lambda x: x >= 0, my_list)

non_negative_list = list(non_negative_filter_object)

print(non_negative_list)  # [3, 2, 0, 14]
```

现在，你应该对如何将函数作为参数传递给其他函数，以及为什么 lambda 表达式有用且重要，有了直观的理解。

## 装饰器

装饰器代表一个接受另一个函数作为参数的函数。

我们可以将其视为一种动态方式，用于改变函数、方法或类的行为，而无需使用子类。

一旦一个函数作为参数传递给装饰器，它将被修改，然后作为一个新函数返回。

让我们从一个我们想要装饰的基本函数开始：

```python
def reverse_list(input_list):
    return input_list[::-1]
```

在这个例子中，我们只是返回一个反转的列表。

我们也可以编写一个接受另一个函数作为参数的函数：

```python
def reverse_list(input_list):
    return input_list[::-1]

def reverse_input_list(another_function, input_list):
    # 我们将执行委托给 another_function
    return another_function(input_list)

result = reverse_input_list(reverse_list, [1, 2, 3])
print(result)  # [3, 2, 1]
```

我们也可以将一个函数嵌套到另一个函数中：

```python
def reverse_input_list(input_list):
    # reverse_list() 现在是一个局部函数，无法从外部访问
    def reverse_list(another_list):
        return another_list[::-1]

    result = reverse_list(input_list)
    return result  # 返回局部函数的结果

result = reverse_input_list([1, 2, 3])
print(result)  # [3, 2, 1]
```

在这个例子中，`reverse_list()` 现在是一个局部函数，无法在 `reverse_input_list()` 函数的作用域之外调用。

现在我们可以编写我们的第一个装饰器：

```python
def reverse_list_decorator(input_function):
    def function_wrapper():
        returned_result = input_function()
        reversed_list = returned_result[::-1]
        return reversed_list

    return function_wrapper
```

`reverse_list_decorator()` 是一个装饰器函数，它接受另一个函数作为输入。要调用它，我们需要编写另一个函数：

```python
# 我们想要装饰的函数
def get_list():
    return [1, 2, 3, 4, 5]
```

现在我们可以用我们的新函数作为参数来调用装饰器：

```python
decorator = reverse_list_decorator(get_list)  # 这返回一个对函数的引用

result_from_decorator = decorator()  # 这里我们使用括号调用实际的函数

# 我们现在可以在控制台打印结果
print(result_from_decorator)  # [5, 4, 3, 2, 1]
```

这是一个完整的例子：

```python
def reverse_list_decorator(input_function):
    def function_wrapper():
        returned_result = input_function()
        reversed_list = returned_result[::-1]
        return reversed_list

    return function_wrapper

# 我们想要装饰的函数
def get_list():
    return [1, 2, 3, 4, 5]

# 这返回一个对函数的引用
decorator = reverse_list_decorator(get_list)

# 这里我们使用括号调用实际的函数
result_from_decorator = decorator()

# 我们现在可以在控制台打印结果
print(result_from_decorator)  # [5, 4, 3, 2, 1]
```

我们也可以使用注解来调用装饰器。为此，我们在想要调用的装饰器名称前使用 `@` 符号，并将其放在函数名称的正上方：

```python
# 我们想要装饰的函数
@reverse_list_decorator  # 装饰器函数的注解
def get_list():
    return [1, 2, 3, 4, 5]
```

现在，我们可以简单地调用函数 `get_list()`，装饰器将被应用到它上面：

```python
result_from_decorator = get_list()

print(result_from_decorator)  # [5, 4, 3, 2, 1]
```

## 堆叠装饰器

我们也可以为一个函数使用多个装饰器。它们的执行顺序是从上到下，这意味着首先定义的装饰器先被应用，然后是第二个，依此类推。

让我们做一个简单的实验，将我们在上一节中定义的同一个装饰器应用两次。

让我们首先理解这意味着什么。

所以我们首先调用装饰器来反转一个列表：

[1, 2, 3, 4, 5] 变为 [5, 4, 3, 2, 1]

然后我们再次应用它，但这次使用的是上一次调用装饰器返回的结果：

[5, 4, 3, 2, 1] ⇒ [1, 2, 3, 4, 5]

换句话说，反转一个列表，然后再次反转那个反转后的列表，将返回列表的原始顺序。

让我们用装饰器来看看这个：

```python
@reverse_list_decorator
@reverse_list_decorator
def get_list():
    return [1, 2, 3, 4, 5]

result = get_list()

print(result)  # [1, 2, 3, 4, 5]
```

让我们用另一个例子来解释这个。
让我们实现另一个装饰器，只返回大于 1 的数字。然后我们想用我们现有的装饰器反转那个返回的列表。

```python
def positive_numbers_decorator(input_list):
    def function_wrapper():
        # 只获取大于 0 的数字
        numbers = [number for number in input_list() if number > 0]
        return numbers

    return function_wrapper
```

现在我们可以调用这个装饰器和我们已经实现的另一个装饰器：

```python
@positive_numbers_decorator
@reverse_list_decorator
def get_list():
    return [1, -2, 3, -4, 5, -6, 7, -8, 9]

result = get_list()
print(result)  # [9, 7, 5, 3, 1]
```

这是一个完整的例子：

```python
def reverse_list_decorator(input_function):
    def function_wrapper():
        returned_result = input_function()
        reversed_list = returned_result[::-1]  # 反转列表
        return reversed_list

    return function_wrapper

# 第一个装饰器
def positive_numbers_decorator(input_list):
    def function_wrapper():
        # 只获取大于 0 的数字
        numbers = [number for number in input_list() if number > 0]
        return numbers

    return function_wrapper

# 我们想要装饰的函数

@positive_numbers_decorator
@reverse_list_decorator
def get_list():
    return [1, -2, 3, -4, 5, -6, 7, -8, 9]

result = get_list()
print(result)  # [9, 7, 5, 3, 1]
```

## 在装饰器函数中传递参数

我们也可以向装饰器函数传递参数：

```python
def add_numbers_decorator(input_function):
    def function_wrapper(a, b):
        result = 'The sum of {} and {} is {}'.format(
            a, b, input_function(a, b))  # 使用参数调用输入函数
        return result
    return function_wrapper

@add_numbers_decorator
def add_numbers(a, b):
    return a + b

print(add_numbers(1, 2))  # The sum of 1 and 2 is 3
```

## 内置装饰器

Python 自带多个内置装饰器，例如 `@classmethod`、`@staticmethod`、`@property` 等。这些将在下一章中介绍。

## 总结

总之，Python 是一种编写函数的优秀语言，因为它们易于编写。

Lambda 函数是在 Python 中创建小型、简洁函数的好方法。当你不需要一个完整的函数，或者只是想测试一段代码片段时，它们非常完美。

Python 装饰器是提高代码可读性和可维护性的好方法。它们允许你将代码模块化并使其更有条理。此外，它们可以用于执行各种任务，例如日志记录、异常处理和测试。因此，如果你正在寻找一种清理 Python 代码的方法，可以考虑使用装饰器。

# 面向对象编程

如果你去当地商店买一块饼干，你得到的将是许多其他副本中生产出来的一块饼干。

工厂里有一个饼干模具，用于生产大量饼干，然后分发到不同的商店，最后准备好供应给最终客户。

我们可以将那个饼干模具视为一个设计一次然后重复使用的蓝图。这个蓝图也用于计算机编程。

一个用于创建无数其他副本的蓝图被称为**类**。我们可以将类想象成一个名为**饼干**、**工厂**、**建筑**、**书**、**铅笔**等的类。我们可以使用**铅笔**类作为蓝图来创建我们想要的任意数量的实例，我们称之为对象。

换句话说，蓝图是用作饼干模具的类，而在不同商店供应的饼干是<u>对象</u>。

**面向对象编程**代表了一种将程序组织为类和对象的方式。类用于创建对象。对象彼此交互。

我们并不为每一个存在的对象使用完全相同的蓝图。有一个用于生产书的蓝图，另一个用于生产铅笔，等等。我们需要根据属性和功能对它们进行分类。

从铅笔类创建的对象可以具有颜色类型、制造商、特定厚度等。这些是**属性**。

一个*pencil*对象也可以<u>书写</u>，这代表了它的功能，或者说它的**方法**。

我们在不同的编程语言中使用类和对象，包括Python。

让我们看看一个非常基础的<u>Bicycle</u>类在Python中是什么样子：

```python
class Bicycle:
    pass
```

我们使用了关键字<u>class</u>来表明我们即将开始编写一个类，然后输入类的名称。

我们添加了<u>pass</u>部分，因为我们希望Python解释器不要因为没有继续编写属于这个类的剩余代码部分而向我们抛出错误。

现在，如果我们想从这个<u>Bicycle</u>类创建新对象，我们只需写下对象的名称（可以是任何你想要的变量名），并使用用于创建新对象的构造方法<u>Bicycle()</u>来初始化它：

```python
favorite_bike = Bicycle()
```

在这种情况下，<u>favorite_bike</u>是一个从<u>Bicycle</u>类创建的对象。它获得了Bicycle类的所有功能和属性。

我们可以丰富我们的<u>Bicycle</u>类，包含额外的属性，这样我们就可以拥有定制的<u>自行车</u>，以满足我们的需求。

为此，我们可以定义一个名为*init*的构造方法，如下所示：

```python
class Bicycle:
    def __init__(self, manufacturer, color, is_mountain_bike):
        self.manufacturer = manufacturer
        self.color = color
        self.is_mountain_bike = is_mountain_bike
```

注意方法名init前后下划线的用法。它们是给Python解释器的指示，表明该方法是一个特殊方法。

这是一个不返回任何东西的方法。将其定义为类的第一个方法是一个好习惯，这样其他开发人员也能看到它位于特定的行。

现在，如果我们想使用这个自行车蓝图创建新对象，我们可以简单地写：

```python
bike = Bicycle("Connondale", "grey", True)
```

我们为这辆自行车提供了自定义参数，并将它们传递给构造方法，以便我们获得一辆具有这些特定属性的新自行车。正如你可能看出的，我们正在创建一辆灰色的、Connondale品牌的山地自行车。

我们也可以通过使用可选参数从类创建对象，如下所示：

```python
class Bicycle:
    # 以下所有属性都是可选的
    def __init__(self, manufacturer=None, color='grey', is_mountain_bike=False):
        self.manufacturer = manufacturer
        self.color = color
        self.is_mountain_bike = is_mountain_bike
```

现在我们刚刚创建了这个具有这些属性的对象，这些属性目前在类的作用域之外是不可访问的。这意味着我们从Bicycle类创建了这个新对象，但其对应的属性是不可访问的。要访问它们，我们可以实现一些方法来帮助我们访问。

为此，我们将定义getter和setter，它们代表用于获取和设置对象属性值的方法。我们将使用一个名为@property的注解来帮助我们。

让我们通过代码来看：

```python
class Bicycle:
    def __init__(self, manufacturer, color, is_mountain_bike):
        self._manufacturer = manufacturer
        self._color = color
        self._is_mountain_bike = is_mountain_bike

    @property
    def manufacturer(self):
        return self._manufacturer

    @manufacturer.setter
    def manufacturer(self, manufacturer):
        self._manufacturer = manufacturer


bike = Bicycle("Connondale", "Grey", True)

print(bike.manufacturer)  # Connondale
```

我们可以为类的所有属性编写getter和setter：

```python
class Bicycle:
    def __init__(self, manufacturer, color, is_mountain_bike):
        self._manufacturer = manufacturer
        self._color = color
        self._is_mountain_bike = is_mountain_bike

    @property
    def manufacturer(self):
        return self._manufacturer

    @manufacturer.setter
    def manufacturer(self, manufacturer):
        self._manufacturer = manufacturer

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, color):
        self._color = color

    @property
    def is_mountain_bike(self):
        return self._is_mountain_bike

    @is_mountain_bike.setter
    def is_mountain_bike(self, is_mountain_bike):
        self.is_mountain_bike = is_mountain_bike

bike = Bicycle("Connondale", "Grey", True)
```

现在我们已经定义了它们，我们可以像调用属性一样调用这些getter方法：

```python
print(bike.manufacturer)  # Connondale
print(bike.color)  # Grey
print(bike.is_mountain_bike)  # True
```

我们也可以通过简单地输入对象名称和我们想要更改内容的属性来修改我们最初用于任何属性的值：

```python
bike.is_mountain_bike = False
bike.color = "Blue"
bike.manufacturer = "Trek"
```

我们的类也可以有其他方法，而不仅仅是getter和setter。

让我们在Bicycle类内部定义一个方法，然后我们可以从从该类创建的任何对象中调用它：

```python
class Bicycle:
    def __init__(self, manufacturer, color, is_mountain_bike):
        self._manufacturer = manufacturer
        self._color = color
        self._is_mountain_bike = is_mountain_bike

    def get_description(self):
        desc = "This is a " + self._color + " bike of the brand " + self._manufacturer
        return desc
```

我们创建了一个非常简单的方法，其中我们准备一个字符串作为我们正在创建的对象的属性的结果。然后我们可以像调用任何其他方法一样调用这个方法。

让我们看看实际效果：

```python
bike = Bicycle("Connondale", "Grey", True)

print(bike.get_description())  # This is a Grey bike of the brand Connondale
```

## 方法

方法类似于我们之前已经介绍过的函数。

简而言之，我们将一些语句分组在一个称为方法的代码块中，在其中执行一些我们期望执行多次且不想一遍又一遍编写的操作。最后，我们可能根本不返回任何结果。

Python中有三种类型的方法：

- 实例方法
- 类方法
- 静态方法

让我们简要谈谈方法的整体结构，然后更深入地了解每种方法类型的细节。

## 参数

方法的参数使我们能够传递动态值，这些值在执行方法内部的语句时可以被考虑在内。

`return`语句代表该方法中将要执行的最后一条语句，因为它是Python解释器停止执行任何其他行并返回值的指示器。

## self参数

Python中方法的第一个参数是`self`，这也是方法和函数之间的区别之一。它代表对其所属对象的引用。如果在声明时未将其指定为方法的第一个参数，则第一个参数将被视为对对象的引用。

我们只在声明方法时编写它，但在使用对象作为调用者调用该特定方法时不需要包含它。它不需要命名为`self`，但这是全世界编写Python代码的开发人员广泛实践的惯例。

让我们在`Bicycle`类内部定义一个实例方法，然后我们可以从从该类创建的任何对象中调用它：

```python
class Bicycle:
    def __init__(self, manufacturer, color, is_mountain_bike):
        self._manufacturer = manufacturer
        self._color = color
        self._is_mountain_bike = is_mountain_bike

    def get_description(self):
        desc = "This is a " + self._color + " bike of the brand " + self._manufacturer
        return desc
```

我们创建了一个非常简单的方法，其中我们准备一个字符串作为我们正在创建的对象的属性的结果。然后我们可以像调用任何其他方法一样调用这个方法。

```python
bike = Bicycle("Connondale", "Grey", True)

print(bike.get_description())  # 这是一辆品牌为Connondale的灰色自行车
# 调用get_description()方法时，我们没有传递任何参数，因为根本不需要包含self
```

## 类方法

到目前为止，我们已经介绍了实例方法，即可以通过对象调用的方法。

类方法是可以通过类名调用的方法，并且完全不需要创建任何新对象即可访问。

由于它是一种特定类型的方法，我们需要告诉Python解释器它确实有所不同。我们通过在语法上做出改变来实现这一点。

我们在类方法上方使用注解`@classmethod`，并使用`cls`，类似于实例方法中`self`的用法。`cls`只是引用调用该方法的类的约定方式，并非必须使用这个名称。

让我们声明我们的第一个类方法：

```python
class Article:
    blog = 'https://www.python.org/'

    # 当类的实例被创建时，init方法会被调用
    def __init__(self, title, content):
        self.title = title
        self.content = content

    @classmethod
    def get_blog(cls):
        return cls.blog
```

现在让我们调用刚刚声明的这个类方法：

```python
print(Article.get_blog())  # https://www.python.org/
```

请注意，调用`get_blog()`方法时，我们不需要编写任何参数。另一方面，当我们声明方法和实例方法时，我们应该始终包含至少一个参数。

## 静态方法

这些方法与类变量或实例变量没有直接关系。它们可以被视为实用函数，旨在帮助我们处理调用时传递的参数。

我们可以通过类名和该方法所在类创建的对象来调用它们。这意味着它们不需要第一个参数与调用它们的对象或类相关，而这在实例方法中使用参数`self`和类方法中使用`cls`时是必需的。

调用它们时可以使用的参数数量没有限制。

要创建它，我们需要使用`@staticmethod`注解。

让我们创建一个静态方法：

```python
class Article:
    blog = 'https://www.python.org/'

    # 当类的实例被创建时，init方法会被调用
    def __init__(self, title, content):
        self.title = title
        self.content = content

    @classmethod
    def get_blog(cls):
        return cls.blog

    @staticmethod
    def print_creation_date(date):
        print(f'The blog was created on {date}')

article = Article('First Article', 'This is the first article')

# 使用对象调用静态方法
article.print_creation_date('2022-07-18')  # The blog was created on 2022-07-18

# 使用类名调用静态方法
Article.print_creation_date('2022-07-21')  # The blog was created on 2022-07-21
```

静态方法不能修改类或实例属性。它们应该像实用函数一样。

如果我们尝试修改类，将会得到错误：

```python
class Article:
    blog = 'https://www.python.org/'

    # 当类的实例被创建时，init方法会被调用
    def __init__(self, title, content):
        self.title = title
        self.content = content

    @classmethod
    def get_blog(cls):
        return cls.blog

    @staticmethod
    def set_title(self, date):
        self.title = 'A random title'
```

如果我们现在尝试调用这个静态方法，将会得到一个错误：

```python
# 使用类名调用静态方法
Article.set_title('2022-07-21')
```

```
TypeError: set_title() missing 1 required positional argument: 'date'
```

这是因为静态方法没有任何对`self`的引用，因为它们与对象或类没有直接关系，因此不能修改属性。

## 访问修饰符

在创建类时，我们可以限制对某些属性和方法的访问，使它们不容易被访问。

我们有公共和私有访问修饰符。
让我们看看这两种。

### 公共属性

公共属性是从类内部和外部都可以访问的属性。

在Python中，默认情况下所有属性和方法都是公共的。如果我们希望它们是私有的，我们需要明确指定。

让我们看一个公共属性的例子：

```python
class Bicycle:
    def __init__(self, manufacturer, color, is_mountain_bike):
        self.manufacturer = manufacturer
        self.color = color
        self.is_mountain_bike = is_mountain_bike

    def get_manufacturer(self):
        return self.manufacturer
```

在前面的代码块中，`color`和`get_manufacturer()`都可以从类外部访问，因为它们是公共的，可以从类内部和外部访问：

```python
bike = Bicycle("Connondale", "Grey", True)

print(bike.color)  # Grey
print(bike.get_manufacturer())  # Connondale
```

### 私有属性

私有属性只能直接从类内部访问。

我们可以通过使用双下划线来使属性成为私有属性，如下例所示：

```python
class Bicycle:
    def __init__(self, manufacturer, color, is_mountain_bike, old):
        self.manufacturer = manufacturer
        self.color = color
        self.is_mountain_bike = is_mountain_bike
        self.__old = old  # 这是一个私有属性
```

现在如果我们尝试访问`__old`，将会得到一个错误：

```python
bike = Bicycle("Connondale", "Grey", True, False)

print(bike.__old)  # AttributeError: 'Bicycle' object has no attribute '__old'
```

现在让我们看一个例子，其中我们使用双下划线在方法名前声明私有方法：

```python
class Bicycle:
    def __init__(self, manufacturer, color, is_mountain_bike, old):
        self.manufacturer = manufacturer
        self.color = color
        self.is_mountain_bike = is_mountain_bike
        self.__old = old  # 这是一个私有属性

    def __get_old(self):  # 这是一个私有方法
        return self.__old
```

现在，如果我们想从类外部调用这个私有方法，将会抛出一个错误：

```python
bike = Bicycle("Connondale", "Grey", True, False)

print(bike.__get_old())  # AttributeError: 'Bicycle' object has no attribute '__get_old'
```

在Python中，拥有私有变量并不是一种常见的做法。然而，开发者可能认为有必要限制访问，以防止特定变量被无意访问和错误修改。

## 信息隐藏

当你外出使用咖啡机时，并不期望你了解那台机器背后的所有工程细节。你的车也是如此。当你坐在驾驶座上时，你不会分析和理解汽车每个部件的所有细节。你对它们有一些基本的了解，但除此之外，你驾驶得更加自由。

这可以被视为对局外人访问的限制，这样他们就不必担心内部发生的确切细节。

我们也可以在Python中做到这一点。

到目前为止，我们已经看到了面向对象编程的基础模块，例如类和对象。

类是用于创建称为实例的蓝图。我们可以使用不同类的对象相互交互，构建一个健壮的程序。

当我们编写自己的程序时，我们可能需要不让每个人都知道我们类的所有细节。因此，我们可以限制对它们的访问，这样某些属性就不太可能被无意访问和错误修改。

为了帮助我们做到这一点，我们隐藏类的部分内容，并简单地提供一个关于类内部工作原理细节较少的接口。
```

我们可以通过两种方式隐藏数据：

1.  封装
2.  抽象

让我们从封装开始。

## 封装

封装并非Python独有的特殊概念。其他编程语言也使用它。

简而言之，我们可以将其定义为将数据和方法绑定在一个类中。然后使用这个类来创建对象。

我们通过使用`private`访问修饰符来封装类，这些修饰符可以限制对这些属性的直接访问。这可以限制控制。

然后，我们应该编写公共方法，以便向外界提供访问。

这些方法被称为`getters`和`setters`。

**getter**方法是用于获取属性值的方法。

**setter**是用于设置属性值的方法。

让我们首先定义一个`getter`和一个`setter`方法，用于获取值：

```python
class Smartphone:
    def __init__(self, type=None):  # defining initialize for case of no argument
        self.__type = type  # setting the type here in the beginning when the object is created

    def set_type(self, value):
        self.__type = value

    def get_type(self):
        return (self.__type)
```

现在，让我们使用这个类来设置类型并获取类型：

```python
smartphone = Smartphone('iPhone')  # we are setting the type using the constructor method

# getting the value of the type
print(smartphone.get_type())  # iPhone

# Changing the value of the type
smartphone.set_type('Samsung')

# getting the new value of the type
print(smartphone.get_type())  # Samsung
```

到目前为止，我们所做的就是设置并读取从Smartphone类创建的对象的私有属性的值。
我们也可以使用`@property`注解来帮助我们定义getter和setter。
让我们通过代码来看看：

```python
class Bicycle:
    def __init__(self, manufacturer, color):
        self._manufacturer = manufacturer
        self._color = color

    @property
    def manufacturer(self):
        return self._manufacturer

    @manufacturer.setter
    def manufacturer(self, manufacturer):
        self._manufacturer = manufacturer

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, color):
        self._color = color

bike = Bicycle("Connondale", "Grey")
```

现在我们已经定义了它们，我们可以像调用属性一样调用这些getter方法：

```python
print(bike.manufacturer)  # Connondale
print(bike.color)  # Grey
```

我们也可以通过简单地输入对象名称和要修改的属性来修改最初为任何属性设置的值：

```python
bike.is_mountain_bike = False
bike.color = "Blue"
```

我们的类也可以有其他方法，而不仅仅是getter和setter。

让我们在Bicycle类内部定义一个方法，然后可以从我们从该类创建的任何对象中调用它：

```python
class Bicycle:
    def __init__(self, manufacturer, color, is_mountain_bike):
        self._manufacturer = manufacturer
        self._color = color
        self._is_mountain_bike = is_mountain_bike

    def get_description(self):
        desc = "This is a " + self._color + " bike of the brand " + self._manufacturer
        return desc
```

我们创建了一个非常简单的方法，其中我们正在准备一个字符串作为我们正在创建的对象的属性的结果。然后我们可以像调用任何其他方法一样调用这个方法。

让我们看看实际效果：

```python
bike = Bicycle("Connondale", "Grey", True)

print(bike.get_description())  # This is a Grey bike of the brand Connondale
```

## 但为什么我们需要封装？

这看起来相当有前途和花哨，但你可能还不太明白。你可能会觉得你需要额外的理由来解释为什么需要这种类型的隐藏。

为了说明这一点，让我们再看一个类，其中我们有一个名为salary的私有属性。假设我们不关心封装，我们只是想快速构建一个类，并在我们的项目中用于我们的会计客户。

假设我们有以下类：

```python
class Employee:
    def __init__(self, name=None, email=None, salary=None):
        self.name = name
        self.email = email
        self.salary = salary
```

现在，让我们创建一个新的员工对象并相应地初始化其属性：

```python
# We are creating an object
betim = Employee('Betim', 'betim@company.com', 5000)

print(betim.salary)  # 5000
```

由于salary没有以任何方式受到保护，我们可以毫无问题地为这个新对象设置新的salary：

```python
betim.salary = 25000

print(betim.salary)  # 25000
```

正如我们所看到的，这个人的薪水是之前的五倍，而没有经过任何类型的评估或面试。事实上，这发生在几秒钟之内。这可能会严重打击公司的预算。

我们显然不想这样做。我们希望限制对salary属性的访问，使其不能从其他地方被调用。我们可以通过在属性名称前使用双下划线来实现这一点，如下所示：

```python
class Employee:
    def __init__(self, name=None, email=None, salary=None):
        self.__name = name
        self.__email = email
        self.__salary = salary
```

让我们创建一个新对象：

```python
# We are creating an object
betim = Employee('Betim', 'betim@company.com', 1000)
```

现在，如果我们尝试访问其属性，我们无法这样做，因为它们是私有属性：

```python
print(betim.salary)  # 1000
```

尝试访问任何属性都会导致错误：

```
AttributeError: 'Employee' object has no attribute 'salary'
```

我们可以简单地实现一个返回属性的方法，但我们不提供任何方式让某人偷偷增加他们的薪水：

```python
class Employee:
    def __init__(self, name=None, email=None, salary=None):
        self.__name = name
        self.__email = email
        self.__salary = salary

    def get_info(self):
        return self.__name, self.__email, self.__salary
```

现在，我们可以访问由这个类创建的对象的信息：

```python
# We are creating an object
betim = Employee('Betim', 'betim@company.com', '5000')

print(betim.get_info())  # ('Betim', 'betim@company.com', '5000')
```

总之，封装帮助我们保护对象的属性，并以受控的方式提供访问。

## 继承

在现实生活中，我们可以与其他人共享许多特征。

我们都需要吃饭、喝水、工作、睡觉、移动等等。这些以及许多其他行为和特征在全世界数十亿人中共享。它们不是我们这一代人独有的东西。这些特征已经存在了许多代，我们甚至没有机会见过。

这也将在未来世代中持续存在。

我们也可以在计算机编程中使用**继承**在我们自己实现的对象和类之间共享某些特征。这包括属性和方法。

让我们想象一下，我们有一个名为Book的类。它应该包含标题、作者、页数、类别、ISBN等。我们将保持类简单，只使用两个属性：

```python
class Book:
    def __init__(self, title, author):
        self.title = title
        self.author = author

    def get_short_book_paragraph(self):
        short_paragraph = "This is a short paragraph of the book."
        return short_paragraph
```

现在，我们可以从这个类创建一个对象并访问它：

```python
first_book = Book("Atomic Habits", "James Clear")

print(first_book.title)  # Atomic Habits
print(first_book.author)  # James Clear
print(first_book.get_short_book_paragraph())  # This is a short paragraph of the book.
```

现在让我们创建一个Book类的子类，它继承自Book类的属性和方法，但还有另一个额外的方法叫做get_book_description()：

## super() 函数

有一个特殊的函数叫做 `super()`，它可以在子类中使用，用来引用其父类，而无需写出父类的确切名称。

我们通常在初始化方法中使用它，或者在调用父类的属性或方法时使用。

让我们通过示例来了解这三种用法。

### 在初始化方法中使用 super()

我们可以在子类的构造函数方法中使用 `super()`，甚至可以调用父类的构造函数：

```python
class Animal():
    def __init__(self, name, age):
        self.name = name
        self.age = age


class Cat(Animal):
    def __init__(self, name, age):
        super().__init__(name, age)  # 调用父类构造函数
        self.health = 100  # 初始化一个父类中没有的新属性
```

我们也可以用父类的名称来代替 `super()`，效果是一样的：

```python
class Animal():
    def __init__(self, name, age):
        self.name = name
        self.age = age


class Cat(Animal):
    def __init__(self, name, age):
        Animal.__init__(name, age)  # 调用父类构造函数
        self.health = 100  # 初始化一个父类中没有的新属性
```

即使改变子类构造函数内部的代码顺序，也不会导致任何错误。

### 使用 super() 访问父类的类属性

我们可以使用 `super()` 来访问父类的类属性，这在父类和子类使用相同名称的属性时尤其有用。

为了演示这一点，假设我们有一个名为 `call_name` 的类属性，它同时存在于父类和子类中。我们希望从父类和子类中都能访问这个变量。

为此，我们只需写出 `super()` 然后加上变量名：

```python
class Producer:  # 父类
    name = 'Samsung'


class Seller(Producer):  # 子类
    name = 'Amazon'

    def get_product_details(self):
        # 调用父类中的变量
        print("Producer:", super().name)

        # 调用子类中的变量
        print("Seller:", self.name)
```

现在，如果我们调用 `get_product_details()` 方法，控制台将打印出以下内容：

```python
seller = Seller()

seller.get_product_details()

# Producer: Samsung
# Seller: Amazon
```

### 使用 super() 调用父类的方法

我们同样可以使用 `super()` 来调用父类中的方法。

```python
class Producer:  # 父类
    name = 'Samsung'

    def get_details(self):
        return f'Producer name: {self.name}'

class Seller(Producer):  # 子类
    name = 'Amazon'

    def get_details(self):
        # 调用父类中的方法
        print(super().get_details())

        # 调用子类中的变量
        print(f'Seller name: {self.name}')

seller = Seller()
seller.get_details()

# Producer name: Amazon
# Seller name: Amazon
```

这就是关于 `super()` 你需要知道的全部内容。

## 继承的类型

根据父类和子类之间的关系，我们可以有不同类型的继承：

- 1. 单继承
- 2. 多级继承
- 3. 层次继承
- 4. 多重继承
- 5. 混合继承

### 1. 单继承

我们可以有一个只从另一个类继承的类：

```python
class Animal:
    def __init__(self):
        self.health = 100

    def get_health(self):
        return self.health


class Cat(Animal):
    def __init__(self, name):
        super().__init__()
        self.health = 150
        self.name = name

    def move(self):
        print("Cat is moving")

cat = Cat("Cat")

# 调用父类中的方法
print(cat.get_health())  # 150

# 调用子类中的方法
cat.move()  # Cat is moving
```

### 2. 多级继承

这是另一种继承类型，其中一个类从另一个类继承，而另一个类又从第三个类继承：类 A 继承自类 B，类 B 继承自类 C。

让我们在 Python 中实现它：

```python
class Creature:
    def __init__(self, alive):
        self.alive = alive

    def is_it_alive(self):
        return self.alive


class Animal(Creature):
    def __init__(self):
        super().__init__(True)
        self.health = 100

    def get_health(self):
        return self.health


class Cat(Animal):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def move(self):
        print("Cat is moving")

cat = Cat("Cat")

# 调用父类的父类中的方法
print(cat.is_it_alive())

# 调用父类中的方法
print(cat.get_health())  # 150

# 调用子类中的方法
cat.move()  # Cat is moving
```

### 3. 层次继承

当我们从同一个父类派生出多个子类时，就形成了层次继承。这些子类都从父类继承：

```python
class Location:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_location(self):
        return self.x, self.y

class Continent(Location):
    pass

class Country(Location):
    pass

continent = Continent(0, 0)
print(continent.get_location())  # (0, 0)

country = Country(10, 30)
print(country.get_location())  # (10, 30)
```

### 4. 多重继承

我们可以有另一种继承类型，即多重继承，它可以帮助我们同时从多个类继承。

假设我们有一个名为 `Date` 的类和另一个名为 `Time` 的类。

然后我们可以实现另一个类，同时继承这两个类：

```python
class Date:
    date = '2022-07-23'  # 硬编码的日期

    def get_date(self):
        return self.date


class Time:
    time = '20:20:20'  # 硬编码的时间

    def get_time(self):
        return self.time


class DateTime(Date, Time):  # 同时继承两个类
    def get_date_time(self):
        return self.get_date() + ' ' + self.get_time()  # 从其父类获取方法
```

## 5. 混合继承

混合继承是多重继承和多级继承的组合：

```python
class Vehicle:
    def print_vehicle(self):
        print('Vehicle')


class Car(Vehicle):
    def print_car(self):
        print('Car')


class Ferrari(Car):
    def print_ferrari(self):
        print('Ferrari')


class Driver(Ferrari, Car):
    def print_driver(self):
        print('Driver')
```

现在，如果我们从 `Driver` 类创建一个对象，我们就可以调用所有类中的所有方法：

```python
driver = Driver()

# 调用子类中的所有方法
driver.print_vehicle()  # Vehicle
driver.print_car()  # Car
driver.print_ferrari()  # Ferrari
driver.print_driver()  # Driver
```

## 多态

这是面向对象编程中的另一个重要概念，指的是一个对象能够表现出不同形式并调用不同行为的可能性。

一个使用多态的内置函数示例是 `len()` 方法，它既可以用于字符串，也可以用于列表：

```python
print(len('Python'))  # 6

print(len([2, 3, -43]))  # 3
```

我们可以用另一个名为 `House` 的类来举例。我们可以有不同的子类，它们从该超类继承方法和属性，例如 `Condo`、`Apartment`、`SingleFamilyHouse`、`MultiFamilyHouse` 等类。

假设我们想在 `House` 类中实现一个用于获取面积的方法。

每种类型的住宅都有不同的大小，因此每个子类都应该有不同的实现。

现在我们可以在子类中定义如下方法：

- getAreaOfCondo()
- getAreaOfApartment()
- getAreaOfSingleFamilyHouse()
- getAreaOfMultiFamilyHouse()

这将迫使我们记住每个子类的名称，这可能很繁琐，并且在调用时也容易出错。
然而，有一个更简单的方法可以使用，它来自多态。
我们可以通过方法和继承来实现多态。
让我们首先看看如何使用方法来实现多态。

## 使用方法实现多态

假设我们有两个类，即 `Condo` 和 `Apartment`。它们都有一个返回值的 `get_area()` 方法。
每个类都将有自己的自定义实现。
现在，将被调用的方法取决于对象的类类型：

```python
class Condo:
    def __init__(self, area):
        self.area = area

    def get_area(self):
        return self.area


class Apartment:
    def __init__(self, area):
        self.area = area

    def get_area(self):
        return self.area
```

让我们从这些类创建两个对象：

```python
condo = Condo(100)

apartment = Apartment(200)
```

现在，我们可以将它们都放入一个列表中，并为两个对象调用相同的方法：

```python
places_to_live = [condo, apartment]

for place in places_to_live:
    print(place.get_area())  # 两个对象使用相同的方法
```

执行后，我们将在控制台中看到以下内容：

```
# 100
# 200
```

这就是使用方法实现的多态。

## 使用继承实现多态

我们不仅可以从超类调用方法。我们还可以为每个子类使用相同的方法名，但有不同的实现。

让我们首先定义一个超类：

```python
class House:
    def __init__(self, area):
        self.area = area

    def get_price(self):
        pass
```

然后让我们实现超类 `House` 的子类 `Condo` 和 `Apartment`：

```python
class House:
    def __init__(self, area):
        self.area = area

    def get_price(self):
        pass


class Condo(House):
    def __init__(self, area):
        self.area = area

    def get_price(self):
        return self.area * 100


class Apartment(House):
    def __init__(self, area):
        self.area = area

    def get_price(self):
        return self.area * 300
```

正如我们所看到的，两个子类都有 `get_price()` 方法，但实现不同。

我们现在可以从子类创建新对象并调用此方法，该方法将根据调用它的对象进行*多态*：

```python
condo = Condo(100)

apartment = Apartment(200)

places_to_live = [condo, apartment]

for place in places_to_live:
    print(place.get_price())
```

执行后，我们将在控制台中看到以下内容：

```
# 10000
# 60000
```

这是多态的另一个例子，其中我们有一个具有相同名称的方法的特定实现。

## 导入

使用像 Python 这样流行语言的主要好处之一是其大量的库，你可以使用并从中受益。

世界各地的许多开发者都非常慷慨地分享他们的时间和知识，并发布了许多非常有用的库，这些库可以在我们的专业工作以及我们可能出于兴趣而做的个人项目中为我们节省大量时间。

以下是一些具有非常有用方法的模块，你可以立即开始在项目中使用：

- time：时间访问和转换
- csv：CSV 文件读写
- math：数学函数
- email：创建、发送和处理电子邮件
- urllib：处理 URL

要导入一个或多个模块，我们只需要写 `import`，然后是我们想要导入的模块名称。

让我们导入我们的第一个模块：

```python
import os
```

现在，让我们一次导入多个模块：

```python
import os, numbers, math
```

一旦我们导入了一个模块，我们就可以开始使用其中的方法。

```python
import math

print(math.sqrt(81))  # 9.0
```

我们也可以通过为导入的模块指定别名来使用新名称，其中别名是你想要的任何变量名：

```python
import math as math_module_that_i_just_imported

result = math_module_that_i_just_imported.sqrt(4)

print(result)  # 2.0
```

## 限制要导入的部分

有时我们不想导入包含所有方法的整个包，因为我们想避免模块中的方法或变量与我们自己想要实现的方法或变量发生覆盖。

我们使用以下形式指定要导入的部分：

```python
from module import function
```

让我们以仅从 math 模块导入平方根函数为例：

```python
from math import sqrt

print(sqrt(100))  # 10.0
```

## 导入所有内容

我们也可以从一个模块导入所有内容，这可能会带来问题。让我们用一个例子来说明这一点。

假设我们想导入 math 模块中包含的所有内容。我们可以通过使用星号来实现这一点

```python
from math import *  # 星号是导入时包含所有内容的指示符
```

现在，假设我们想声明一个名为 `sqrt` 的变量：

```python
sqrt = 25
```

当我们尝试调用 math 模块中的 `sqrt()` 函数时，我们将得到一个错误，因为解释器将调用我们在上一个代码块中刚刚声明的最新的 `sqrt` 变量：

```python
print(sqrt(100))
```

```
TypeError: 'float' object is not callable
```

## 异常

当我们实现 Python 脚本或进行任何类型的实现时，即使语法正确，我们也会遇到许多抛出的错误。

这些在执行过程中发生的错误称为异常。

我们确实不必放弃，也不必对它们置之不理。我们可以编写处理程序来处理这些情况，以便程序的执行不会停止。

### 常见异常

以下是 Python 中发生的一些最常见的异常，定义取自 [Python 文档](https://docs.python.org/3/library/exceptions.html)：

- **Exception**（这是一个类，作为大多数其他异常类型的超类）
- **NameError** - 当找不到本地或全局名称时引发。
- **AttributeError** - 当属性引用或赋值失败时引发。
- **SyntaxError** - 当解析器遇到语法错误时引发。
- **TypeError** - 当操作或函数应用于不适当类型的对象时引发。关联值是一个字符串，提供有关类型不匹配的详细信息。
- **ZeroDivisionError** - 当除法或取模运算的第二个参数为零时引发。

¹https://docs.python.org/3/library/exceptions.html

## IOError
- **IOError** - 当 I/O 操作（例如 `print` 语句、内置的 `open()` 函数或文件对象的方法）因 I/O 相关原因（如“文件未找到”或“磁盘已满”）失败时引发。
- **ImportError** - 当 `import` 语句未能找到模块定义，或 `from ... import` 未能找到要导入的名称时引发。
- **IndexError** - 当序列下标超出范围时引发。
- **KeyError** - 当映射（字典）的键在现有键集合中未找到时引发。
- **ValueError** - 当内置操作或函数接收到类型正确但值不合适的参数，且该情况未被更精确的异常（如 `IndexError`）描述时引发。

还有许多其他错误类型，但你现在可能不需要了解它们。而且，你也不太可能同时遇到所有类型的错误。

你可以在 [Python 文档](https://docs.python.org/3/library/exceptions.html) 中查看更多异常类型。

## 处理异常

让我们从一个非常简单的例子开始，编写一个故意抛出错误的程序，以便我们随后修复它。

我们将进行一个除以零的操作，这可能是你在学校里见过的：

```
print(5 / 0)
```

如果我们尝试执行它，控制台将显示以下错误：

```
ZeroDivisionError: division by zero
```

如果在任何 Python 程序中出现这种情况，我们应该捕获这个错误并将其包装在 `try/except` 块中。

我们需要在 `try` 块中编写我们预计会抛出错误的代码部分。然后，我们在 `except` 块中捕获这些类型的错误，同时指定我们预期发生的错误类型。

让我们看第一个例子。

让我们看看如何处理这个错误，以便我们也能得知发生了这样的错误：

```python
try:
    5 / 0
except ZeroDivisionError:
    print('You cannot divide by 0 mate!')
```

如你所见，一旦我们到达除以 0 的部分，我们就在控制台打印一条消息。

我们也可以完全省略 `ZeroDivisionError` 部分：

```python
try:
    5 / 0
except:
    print('You cannot divide by 0 mate!')
```

然而，这并不推荐，因为我们在单个 `except` 块中捕获了所有类型的错误，我们不确定捕获了什么类型的错误，这对我们来说可能非常有用。

让我们继续看另一种类型的错误。

让我们尝试使用一个完全未定义的变量：

```python
name = 'User'

try:
    person = name + surname  # surname is not declared
except NameError:
    print('A variable is not defined')
```

在前面的例子中，我们在声明变量 `surname` 之前使用了它，因此将抛出 `NameError`。

让我们继续看另一种可能常见的错误类型。

当我们使用列表时，使用超出范围的索引是一个常见的错误，这意味着使用的索引大于或小于该列表中元素的索引范围。

让我们用一个例子来说明，其中将抛出 `IndexError`：

```python
my_list = [1, 2, 3, 4]

try:
    print(my_list[5])
    # This list only has 4 elements, so its indexes range
    # from 0 to 3
except IndexError:
    print('You have used an index that is out of range')
```

我们也可以在单个 `try` 块中使用多个 `except` 错误：

```python
my_list = [1, 2, 3, 4]

try:
    print(my_list[5])
    # This list only has 4 elements, so its indexes range
    # from 0 to 3
except NameError:
    print('You have used an invalid value')
except ZeroDivisionError:
    print('You cannot divide by zero')
except IndexError:
    print('You have used an index that is out of range')
```

在前面的例子中，我们首先尝试捕获是否有任何使用但未声明的变量。如果发生此错误，那么这个 `except` 块将接管执行流。执行流将在此处停止。

然后，我们尝试检查是否除以零。如果抛出此错误，那么这个 `except` 块将接管执行，其中的所有内容都将被执行。类似地，我们继续处理声明的其余错误。

我们也可以在括号中放入多个错误来捕获多个异常，但这对我们没有帮助，因为我们不知道抛出了什么具体的错误。换句话说，以下方法有效，但不推荐：

```python
my_list = [1, 2, 3, 4]

try:
    print(my_list[5])
    # This list only has 4 elements, so its indexes range
    # from 0 to 3
except (NameError, ZeroDivisionError, IndexError):
    print('A NameError, ZeroDivisionError, or IndexError occurred')
```

## finally

在 `try` 和 `except` 之后，我们可以声明并执行另一个块。这个块以 `finally` 关键字开头，无论是否抛出错误，它都会被执行：

```python
my_list = ['a', 'b']

try:
    print(my_list[0])
except IndexError:
    print('An IndexError occurred')
finally:
    print('The program is ending. This is going to be executed.')
```

如果我们执行前面的代码块，控制台将显示以下内容：

```
a
The program is ending. This is going to be executed.
```

我们通常将希望作为清理工作的代码写在 `finally` 块中。这包括关闭文件、停止与数据库的连接、完全退出程序等。

## try, else, except

我们可以在 `try` 和 `except` 中编写语句，但也可以使用 `else` 块，其中我们可以编写希望在没有抛出错误时执行的代码：

```python
my_list = ['a', 'b']

try:
    print(my_list[0])
except IndexError:
    print('An IndexError occurred')
else:
    print('No error occurred. Congratulations!')
```

如果我们执行上面的代码，控制台将打印以下内容：

```
a
No error occurred. Congratulations!
```

## 总结

这应该足以让你理解异常以及你可以用来处理它们的方法，这样就不会有突然的中断导致你的程序意外失败。

## 后记

本书代表了我为你快速轻松地学习 Python 基础知识所做的尝试。Python 还包含许多其他内容，本书未涵盖，但我们将在此止步。

我希望这能成为你有用的参考。

既然你已经有机会学习如何编写 Python，那就走出去，用你的代码行产生积极的影响。