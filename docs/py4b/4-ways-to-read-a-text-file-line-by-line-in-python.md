# 用 Python 逐行读取文本文件的 4 种方法

> 原文：<https://www.pythonforbeginners.com/files/4-ways-to-read-a-text-file-line-by-line-in-python>

在任何编程语言中，读取文件都是一项必要的任务。无论是数据库文件、图像还是聊天日志，读写文件的能力极大地增强了我们使用 Python 的能力。

在我们创建自动化有声读物阅读器或网站之前，我们需要掌握一些基本知识。毕竟，没有人在爬之前跑过。

更复杂的是，Python 提供了几种读取文件的解决方案。我们将讨论在 Python 中逐行读取文件的最常见的过程。一旦你掌握了用 Python 阅读文件的基本知识，你就能更好地应对未来的挑战。

您会很高兴地了解到 Python 提供了几个用于读取、写入和创建文件的函数。这些功能通过在幕后为我们处理一些工作来简化文件管理。

我们可以使用许多 Python 函数来逐行读取文件。

## 使用 readlines()方法逐行读取文件

我们在 Python 中读取文件的第一种方法是阻力最小的方法: **readlines()** 方法。这个方法将打开一个文件，并把它的内容分成几行。

该方法还返回文件中所有行的列表。我们可以使用 readlines()快速读取整个文件。

例如，假设我们有一个包含公司员工基本信息的文件。我们需要读取这个文件并对数据做些什么。

*employees.txt*
姓名:马库斯·盖伊
年龄:25
职业:网络开发员

姓名:莎莉雨
年龄:31
职业:高级程序员

#### 示例 1:使用 readlines()读取文件

```py
# open the data file
file = open("employees.txt")
# read the file as a list
data = file.readlines()
# close the file
file.close()

print(data) 
```

**输出**

```py
['Name: Marcus Gaye\n', 'Age: 25\n', 'Occupation: Web Developer\n', '\n', 'Name: Sally Rain\n', 'age: 31\n', 'Occupation: Senior Programmer\n']
```

### readline()与 readlines()

与它的对应物不同， **readline()** 方法只返回文件中的一行。realine()方法还会在字符串末尾添加一个尾随换行符。

使用 readline()方法，我们还可以选择指定返回行的长度。如果没有提供大小，将读取整行。

考虑以下文本文件:

*wise_owl.txt*
一只聪明的老猫头鹰住在一棵橡树上。他越看越少说话。他说得越少，听得越多。为什么我们不能都像那只聪明的老鸟一样？

我们可以使用 readline()来获取文本文档的第一行。与 readlines()不同，当我们使用 readline()方法读取文件时，只打印一行。

#### 示例 2:用 readline()方法读取一行

```py
file = open("wise_owl.txt")
# get the first line of the file
line1 = file.readline()
print(line1)
file.close()
```

**输出**

```py
A wise old owl lived in an oak. 
```

readline()方法只检索单行文本。如果需要一次读取所有行，请使用 readline()。

```py
file = open("wise_owl.txt")
# store all the lines in the file as a list
lines = file.readlines()
print(lines)
file.close() 
```

**输出**

一只聪明的老猫头鹰住在橡树里。\n '，'他越看越少说话。他说得越少，听到的越多。“为什么我们不能都像那只聪明的老鸟一样呢？\n"]

## 使用 While 循环读取文件

也可以使用循环来读取文件。使用我们在上一节中创建的同一个 *wise_owl.txt* 文件，我们可以使用 while 循环读取文件中的每一行。

#### 示例 3:使用 while 循环和 readline()读取文件

```py
file = open("wise_owl.txt",'r')
while True:
    next_line = file.readline()

    if not next_line:
        break;
    print(next_line.strip())

file.close() 
```

**输出**

一只聪明的老猫头鹰住在橡树里。他越看越少说话。他说得越少，听到的越多。
为什么我们不能都像那只聪明的老鸟一样呢？

#### 小心无限循环

在使用 while 循环时，需要注意一点。注意要为循环添加一个终止用例，否则你将永远循环下去。考虑下面的例子:

```py
while True:
    print("Groundhog Day!") 
```

执行这段代码会导致 [Python 陷入无限循环](https://www.pythonforbeginners.com/loops/for-while-and-nested-loops-in-python)，打印“土拨鼠日”直到时间的尽头。当编写这样的代码时，总是要提供一种退出循环的方法。

如果你发现你不小心执行了一个无限循环，你可以在终端中点击键盘上的 **Control+C** 来逃离它。

## 在 Python 中读取文件对象

还可以使用 for 循环读取 Python 中的文件。例如，我们的客户给了我们以前客户的地址列表。我们需要使用 Python 来读取数据。

这是客户名单:

*address_list.txt*

山姆·加菲尔德边境路 9805 号。
新泽西州新不伦瑞克，邮编 08901

纽约林登赫斯特第二街 408 号
,邮编:11757

马库斯·盖伊
佛罗里达州洛克里奇市舒布农场街 622 号
，邮编 32955

普鲁登斯·布朗
明尼苏达州查斯卡市阿什利大街 66 号
邮编:55318

每当我们打开一个文件对象时，我们可以使用一个 for 循环，通过使用关键字中的**来读取它的内容。使用**关键字中的**，我们可以循环遍历文件中的行。**

#### 示例 4:使用 for 循环读取文件中的行

```py
# open the file 
address_list = open("address_list.txt",'r')

for line in address_list:
    print(line.strip())

address_list.close() 
```

不幸的是，这个解决方案不适合我们的客户。数据采用 Python 列表的形式非常重要。在继续之前，我们需要将文件分成不同的地址，并将它们存储在一个列表中。

#### 示例 5:读取文件并将内容拆分成一个列表

```py
file = open("address_list.txt",'r')
address_list = []
i = 0
current_address = ""
for line in file:
    # add a new address every three lines
    if i > 2:
        i = 0
        address_list.append(current_address)
        current_address = ""
    else:
        # add the line to the current address
        current_address += line
        i += 1

# use a for-in loop to print the list of addresses
for address in address_list:
    print(address)

file.close() 
```

## 使用上下文管理器读取文件

在任何编程语言中，文件管理都是一个微妙的过程。必须小心处理文件，以防损坏。当打开一个文件时，必须小心确保资源在以后被关闭。

而且在 Python 中一次可以打开多少文件是有限制的。为了避免这些问题，Python 为我们提供了*上下文管理器。*

#### 用块引入

**每当我们在 Python 中打开一个文件时，记住关闭它是很重要的。像 readlines()这样的方法对于小文件来说可以，但是如果我们有一个更复杂的文档呢？使用 Python **和**语句将确保文件得到安全处理。**

*   **with 语句用于安全地访问资源文件。**
*   **Python 在遇到带有块的**时会创建一个新的上下文。****
*   **一旦块执行，Python 自动关闭文件资源。**
*   **上下文与 with 语句具有相同的范围。**

**让我们通过阅读一封保存为文本文件的电子邮件来练习使用 with 语句。**

***email.txt*
*尊敬的客户，
感谢您就您所购买产品的问题联系我们。我们渴望解决您对我们产品的任何疑虑。***

**我们想确保你对你的购买完全满意。为此，我们对所有库存提供 30 天退款保证。只需退回产品，我们将很乐意退还您的购买价格。**

***谢谢，
ABC 公司***

```py
`code example
# open file
with open("email.txt",'r') as email:
    # read the file with a for loop
    for line in email:
        # strip the newline character from the line
        print(line.strip())` 
```

**这次使用 for 循环来读取文件的行。当我们使用上下文管理器时，当文件的处理程序超出范围时，文件会自动关闭。当函数对文件完成时， **with** 语句确保资源得到负责任的处理。**

## **摘要**

**我们已经介绍了在 Python 中逐行读取文件的几种方法。我们已经知道了 **readline()** 和**readline()**方法之间有很大的区别，我们可以使用 for 循环来读取 file 对象的内容。**

**我们还学习了如何使用带有语句的**来打开和读取文件。我们看到了如何创建上下文管理器来使 Python 中的文件处理更加安全和简单。****

**提供了几个例子来说明 Python 中各种形式的文件处理。花点时间研究这些示例，如果有不明白的地方，不要害怕尝试代码。**

**如果你想学习 Python 编程，请点击下面的链接，查看 Python 为初学者提供的更多精彩课程。**

## **相关职位**

*   **使用 [Python try except](https://www.pythonforbeginners.com/error-handling/python-try-and-except) 处理错误**
*   **如何使用 [Python 拆分字符串](https://www.pythonforbeginners.com/dictionary/python-split)提高你的编码技能**
*   **使用 [Python 字符串串联](https://www.pythonforbeginners.com/concatenation/string-concatenation-and-formatting-in-python)以获得更好的可读性**