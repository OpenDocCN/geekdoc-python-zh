# Python 中的列表理解

> 原文：<https://www.pythonforbeginners.com/basics/list-comprehensions-in-python>

作为 Python 程序员，您可能会使用很多列表。虽然我们都是 for 循环(和嵌套 for 循环)的忠实粉丝，但 Python 提供了一种更简洁的方法来处理列表和列表理解。

为了保持代码的优雅和可读性，建议您使用 Python 的理解特性。

列表理解是在 Python 中创建列表的一种强大而简洁的方法，当您使用列表和列表列表时，它变得必不可少。

## 句法

考虑下面的例子:

*列表*中*项目*的 my_new_list = [ *表达式*

从这个例子中可以看出，理解 python 列表需要三个要素。

1.  首先是我们想要执行的表达式。*方括号内的表达式*。
2.  第二个是表达式将处理的对象。*方括号内的项目*。
3.  最后，我们需要一个可迭代的对象列表来构建我们的新列表。*在方括号内列出*。

要理解列表理解，想象一下:你要对列表中的每一项执行一个表达式。表达式将决定最终在输出列表中存储什么项目。

您不仅可以在一行代码中对整个列表执行表达式，而且，正如我们将在后面看到的，还可以以过滤器的形式添加条件语句，这使得处理列表的方式更加精确。

### 关于列表理解的注记

*   列表理解方法是一种创建和管理列表的优雅方式。
*   在 Python 中，列表理解是一种更紧凑的创建列表的方式。
*   比循环更灵活，列表理解通常比其他方法更快。

## 创建范围为()的列表

让我们从使用 Python list comprehensions 创建一个数字列表开始。我们将利用 Python 的 **range()** 方法来创建一个数字列表。

**示例 1:创建具有列表理解的列表**

```py
# construct a basic list using range() and list comprehensions
# syntax
# [ expression for item in list ]
digits = [x for x in range(10)]

print(digits) 
```

**输出**

```py
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

让我们从第一个“x”开始分解这个 python 例子。这是我们的表达方式。它没有做任何事情，因为我们只是记录数字。第二个“x”代表由 **range()** 方法创建的列表中的每一项。

在上面的 python 例子中，我们使用了 **range()** 方法来生成一个数字列表。 [Python 遍历(或循环](https://www.pythonforbeginners.com/loops/for-while-and-nested-loops-in-python))该范围内的每一项，并将该项的副本保存在一个名为 digits 的新列表中。

也许这似乎是多余的？那只是因为你还没有看到理解列表的真正潜力。

## 使用 Python 中的循环和列表理解创建列表

为了更好地说明如何使用列表理解来编写更高效的 Python 代码，我们来看一下并排比较。

在以下示例中，您将看到创建 Python 列表的两种不同技术。第一个是 for 循环。我们将用它来构造一个 2 的幂的列表。

**示例 2:比较 Python 中的列表创建方法**

首先，创建一个列表并遍历它。添加一个表达式，在这个例子中，我们将 x 提高到 2 的幂。

```py
# create a list using a for loop
squares = []

for x in range(10):
    # raise x to the power of 2
    squares.append(x**2)

print(squares) 
```

```py
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

**输出**

使用 list comprehension 可以做同样的事情，但是只需要一部分代码。让我们来看看如何使用列表理解方法创建一个正方形列表。

```py
# create a list using list comprehension
squares = [x**2 for x in range(10)]

print(squares) 
```

即使在这个基本的例子中，很明显列表理解减少了处理列表时完成复杂任务所需的代码。

## 列表的乘法部分

如果我们想用 Python 把列表中的每个数字都乘以 3 会怎么样？我们可以写一个 for 循环并将结果存储在一个新的列表中，或者我们可以使用列表理解。

**例 3:列表综合乘法**

```py
# create a list with list comprehensions
multiples_of_three = [ x*3 for x in range(10) ]

print(multiples_of_three) 
```

**输出**

```py
[0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
```

通过利用列表理解，可以只处理列表的一部分。例如，如果您只想要给定范围内的偶数，您可以使用过滤器找到它们。

向列表理解添加过滤器允许更大的灵活性。通过使用过滤器，我们可以从列表中选择某些项目，同时排除其他项目。这是 Python 中列表的一个高级特性。

```py
even_numbers = [ x for x in range(1,20) if x % 2 == 0]
```

**输出**

```py
[2, 4, 6, 8, 10, 12, 14, 16, 18]
```

## 使用 Python 显示每个单词的首字母

到目前为止，我们已经看到了用 Python 构建数字列表的例子。接下来，让我们尝试使用字符串和列表理解的各种方法来优雅地处理字符串列表。

**示例 4:使用带有字符串的列表理解**

```py
# a list of the names of popular authors
authors = ["Ernest Hemingway","Langston Hughes","Frank Herbert","Toni Morrison",
    "Emily Dickson","Stephen King"]

# create an acronym from the first letter of the author's names
letters = [ name[0] for name in authors ]
print(letters) 
```

**输出**

```py
['E', 'L', 'F', 'T', 'E', 'S']
```

使用简化的表达式，列表理解可以使解决涉及字符串的问题更加容易。这些方法可以节省时间和宝贵的代码行。

**示例 5:分隔字符串中的字符**

```py
# use list comprehension to print the letters in a string
letters = [ letter for letter in "20,000 Leagues Under The Sea"]

print(letters)
```

**输出**

```py
['2', '0', ',', '0', '0', '0', ' ', 'L', 'e', 'a', 'g', 'u', 'e', 's', ' ', 'U', 'n', 'd', 'e', 'r', ' ', 'T', 'h', 'e', ' ', 'S', 'e', 'a']
```

## 使用 Python 的小写/大写转换器

使用 list comprehension 遍历 Python 中的字符串，可以将字符串从小写转换为大写，反之亦然。

利用 Python 的 **lower()** 和 **upper()** 方法，我们将使用列表理解来实现这个常见任务。

**示例 6:改变字母的大小写**

```py
lower_case = [ letter.lower() for letter in ['A','B','C'] ]
upper_case = [ letter.upper() for letter in ['a','b','c'] ]

print(lower_case, upper_case) 
```

**输出**

```py
['a', 'b', 'c'] ['A', 'B', 'C']
```

## 仅打印给定字符串中的数字

另一个有趣的练习是从字符串中提取数字。例如，我们可能有一个姓名和电话号码的数据库。

如果我们能把电话号码和名字分开会很有用。使用列表理解，我们可以做到这一点。

利用 Python 的 **isdigit()** 方法，我们可以从用户数据中提取出电话号码。

**示例 7:使用 isdigit()方法识别字符串中的数字**

```py
# user data entered as name and phone number
user_data = "Elvis Presley 987-654-3210"
phone_number = [ x for x in user_data if x.isdigit()]

print(phone_number) 
```

**输出**

```py
['9', '8', '7', '6', '5', '4', '3', '2', '1', '0']
```

## 使用列表理解解析文件

也可以使用列表理解来读取 Python 中的文件。为了演示，我创建了一个名为 *dreams.txt* 的文件，并粘贴了下面的文本，一首兰斯顿·休斯的短诗。

紧紧抓住梦想，因为如果梦想消亡，生活就像折断翅膀的鸟儿
再也不能飞翔。兰斯顿·休斯

使用列表理解，我们可以遍历文件中的文本行，并将它们的内容存储在一个新的列表中。

**例 8:阅读一首带有列表理解的诗**

```py
# open the file in read-only mode
file = open("dreams.txt", 'r')
poem = [ line for line in file ]

for line in poem:
    print(line) 
```

**输出**

```py
Hold fast to dreams

For if dreams die

Life is a broken-winged bird

That cannot fly.

-Langston Hughs 
```

## 在列表理解中使用函数

到目前为止，我们已经看到了如何使用 list comprehension 使用一些基本的 Python 方法生成列表，如 **lower()** 和 **upper()** 。但是如果我们想使用自己的 Python 函数呢？

我们不仅可以用列表理解来编写自己的函数，还可以添加过滤器来更好地控制语句。

**例 9:为列表理解添加参数**

```py
# list comprehension with functions
# create a function that returns a number doubled
def double(x):
    return x*2

nums = [double(x) for x in range(1,10)]
print(nums) 
```

**输出**

```py
[2, 4, 6, 8, 10, 12, 14, 16, 18]
```

可以使用附加参数从列表中过滤元素。在下面的示例中，只选择了偶数。

```py
# add a filter so we only double even numbers
even_nums = [double(x) for x in range(1,10) if x%2 == 0]
print(even_nums) 
```

**输出**

```py
[4, 8, 12, 16]
```

可以添加其他参数来创建更复杂的逻辑:

```py
nums = [x+y for x in [1,2,3] for y in [10,20,30]]
print(nums) 
```

**输出**

```py
[11, 21, 31, 12, 22, 32, 13, 23, 33]
```

## 最后

希望您已经看到了列表理解的潜力，以及如何使用它们来编写更优雅的 Python 代码。编写紧凑的代码对于维护程序和与团队合作是必不可少的。

学会利用像列表理解这样的高级功能可以节省时间，提高效率。

当你的 Python 代码更加简洁易读时，不仅你的同事会感谢你，而且当你回到一个你几个月都没做的程序，代码是可管理的时候，你也会感谢你自己。

### 相关职位

有兴趣了解更多关于 Python 编程的知识吗？请点击这些链接，获取将对您的 Python 之旅有所帮助的其他资源。

*   了解一个单独的 [Python 注释](https://www.pythonforbeginners.com/comments/comments-in-python)是如何改进程序的。
*   探索 [Python 字符串串联](https://www.pythonforbeginners.com/concatenation/string-concatenation-and-formatting-in-python)