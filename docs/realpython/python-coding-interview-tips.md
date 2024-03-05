# 如何在 Python 编码面试中脱颖而出

> 原文：<https://realpython.com/python-coding-interview-tips/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**Python 编码面试:技巧&最佳实践**](/courses/python-coding-interviews-tips-best-practices/)

您已经通过了与招聘人员的电话沟通，现在是时候展示您知道如何用实际代码解决问题了。无论是 HackerRank 练习、带回家的作业，还是现场白板面试，这都是你证明自己编码面试技巧的时刻。

但是面试不仅仅是为了解决问题:他们也是为了展示你可以写出干净的产品代码。这意味着您对 Python 的内置功能和库有深入的了解。这些知识向公司表明，你可以快速移动，不会仅仅因为你不知道它的存在而复制语言自带的功能。

**注意:**要了解编码面试的情况并学习编码挑战的最佳实践，请查看视频课程[编写并测试 Python 函数:面试实践](https://realpython.com/courses/interview-practice-python-function/)。

在 *Real Python* 上，我们集思广益，讨论了在编码面试中我们总是印象深刻的工具。本文将带您领略这些功能的精华，从 Python 的内置开始，然后是 Python 对[数据结构](https://realpython.com/python-data-structures/)的本地支持，最后是 Python 强大的(但往往不被重视的)标准库。

**在这篇文章中，你将学习如何:**

*   使用`enumerate()`迭代索引和值
*   用`breakpoint()`调试有问题的代码
*   用 f 字符串有效地格式化字符串
*   使用自定义参数对列表进行排序
*   使用生成器而不是列表理解来节省内存
*   查找字典关键字时定义默认值
*   用`collections.Counter`类计数可散列对象
*   使用标准库获得排列和组合列表

**免费奖励:** [掌握 Python 的 5 个想法](https://realpython.com/bonus/python-mastery-course/)，这是一个面向 Python 开发者的免费课程，向您展示将 Python 技能提升到下一个水平所需的路线图和心态。

## 为作业选择正确的内置功能

Python 有一个很大的标准库，但只有一个很小的[内置函数](https://docs.python.org/library/functions.html)库，这些函数总是可用的，不需要导入。每一个都值得一读，但是在你有机会这样做之前，这里有几个内置函数值得你去理解如何使用，以及在其中一些情况下，用什么替代函数来代替。

[*Remove ads*](/account/join/)

### 用`enumerate()`代替`range()` 进行迭代

这种场景在编码面试中出现的次数可能比其他任何场景都多:您有一个元素列表，您需要通过访问索引和值来遍历该列表。

有一个名为 FizzBuzz 的经典编码面试问题可以通过迭代索引和值来解决。在 FizzBuzz 中，给你一个整数列表。你的任务是做以下事情:

1.  用`"fizz"`替换所有能被`3`整除的整数
2.  用`"buzz"`替换所有能被`5`整除的整数
3.  用`"fizzbuzz"`替换所有能被`3`和`5`整除的整数

通常，开发人员会用`range()`来解决这个问题:

>>>

```py
>>> numbers = [45, 22, 14, 65, 97, 72]
>>> for i in range(len(numbers)):
...     if numbers[i] % 3 == 0 and numbers[i] % 5 == 0:
...         numbers[i] = 'fizzbuzz'
...     elif numbers[i] % 3 == 0:
...         numbers[i] = 'fizz'
...     elif numbers[i] % 5 == 0:
...         numbers[i] = 'buzz'
...
>>> numbers
['fizzbuzz', 22, 14, 'buzz', 97, 'fizz']
```

Range 允许你通过索引访问`numbers`的元素，在某些情况下它是一个有用的工具[。但是在这种情况下，如果您希望同时获得每个元素的索引和值，更好的解决方案是使用](https://realpython.com/python-range/) [`enumerate()`](https://realpython.com/python-enumerate/) :

>>>

```py
>>> numbers = [45, 22, 14, 65, 97, 72]
>>> for i, num in enumerate(numbers):
...     if num % 3 == 0 and num % 5 == 0:
...         numbers[i] = 'fizzbuzz'
...     elif num % 3 == 0:
...         numbers[i] = 'fizz'
...     elif num % 5 == 0:
...         numbers[i] = 'buzz'
...
>>> numbers
['fizzbuzz', 22, 14, 'buzz', 97, 'fizz']
```

对于每个元素，`enumerate()`返回一个计数器和元素值。计数器默认为`0`，这也是元素的索引。不想从`0`开始计算吗？只需使用可选的`start`参数来设置偏移量:

>>>

```py
>>> numbers = [45, 22, 14, 65, 97, 72]
>>> for i, num in enumerate(numbers, start=52):
...     print(i, num)
...
52 45
53 22
54 14
55 65
56 97
57 72
```

通过使用`start`参数，我们从第一个索引开始访问所有相同的元素，但是现在我们的计数从指定的整数值开始。

### 使用列表理解代替`map()`和`filter()`

> “我认为去掉 filter()和 map()是相当没有争议的。]"
> 
> — *吉多·范·罗苏姆，Python 的创造者*

他可能错误地认为它没有争议，但是 Guido 有充分的理由想要从 Python 中删除 [`map()`](https://realpython.com/python-map-function/) 和 [`filter()`](https://realpython.com/python-filter-function/) 。一个原因是 Python 支持[列表理解](https://realpython.com/list-comprehension-python/)，它们通常更容易阅读，并支持与`map()`和`filter()`相同的功能。

让我们首先来看看我们是如何构造对`map()`的调用以及等价的列表理解的:

>>>

```py
>>> numbers = [4, 2, 1, 6, 9, 7]
>>> def square(x):
...     return x*x
...
>>> list(map(square, numbers))
[16, 4, 1, 36, 81, 49]

>>> [square(x) for x in numbers]
[16, 4, 1, 36, 81, 49]
```

使用`map()`和列表理解的两种方法返回相同的值，但是列表理解更容易阅读和理解。

现在我们可以对`filter()`及其等价的列表理解做同样的事情:

>>>

```py
>>> def is_odd(x):
...    return bool(x % 2)
...
>>> list(filter(is_odd, numbers))
[1, 9, 7]

>>> [x for x in numbers if is_odd(x)]
[1, 9, 7]
```

就像我们在`map()`中看到的那样，`filter()`和列表理解方法返回相同的值，但是列表理解更容易理解。

来自其他语言的开发者可能不同意列表理解比`map()`和`filter()`更容易阅读，但是根据我的经验，初学者能够更直观地理解列表理解。

不管怎样，在编码面试中使用列表理解很少会出错，因为它会传达出你知道 Python 中最常见的是什么。

[*Remove ads*](/account/join/)

### 用`breakpoint()`代替`print()` 进行调试

通过在代码中添加 [`print()`](https://realpython.com/python-print/) 并查看打印出来的内容，你可能已经调试出了一个小问题。这种方法一开始工作得很好，但是很快就变得很麻烦。另外，在编码面试环境中，你很难希望 [`print()`调用](https://realpython.com/courses/python-print/)贯穿你的代码。

相反，你应该使用一个[调试器](https://realpython.com/python-debug-idle/)。对于重要的 bug，它几乎总是比使用`print()`更快，鉴于调试是编写软件的一个重要部分，它表明你知道如何使用工具，让你在工作中快速开发。

如果您使用的是 Python 3.7，您不需要导入任何东西，只需在代码中您想要进入调试器的位置调用 [`breakpoint()`](https://realpython.com/python37-new-features/#the-breakpoint-built-in) :

```py
# Some complicated code with bugs

breakpoint()
```

调用`breakpoint()`会让你进入 [`pdb`](https://realpython.com/python-debugging-pdb/) ，这是默认的 Python 调试器。在 Python 3.6 和更早的版本中，您可以通过显式导入`pdb`来完成同样的操作:

```py
import pdb; pdb.set_trace()
```

像`breakpoint()`，`pdb.set_trace()`会把你放入`pdb`调试器。只是不太干净，而且更容易记住。

您可能想尝试其他可用的调试器，但是`pdb`是标准库的一部分，所以它总是可用的。无论您喜欢哪种调试器，在您进入编码面试环境之前，尝试一下它们以适应工作流都是值得的。

### 用 f 字符串格式化字符串

Python 有很多不同的方法来处理字符串格式，知道使用什么可能很棘手。事实上，我们在两篇独立的文章中深入探讨了格式化:一篇是关于一般的[字符串格式化](https://realpython.com/python-string-formatting/)，另一篇是专门针对 f 字符串的[。在一次编码面试中，当你(希望)使用 Python 3.6+时，建议的格式化方法是 Python 的 f 字符串。](https://realpython.com/python-f-strings/)

f-strings 支持使用[字符串格式化迷你语言](https://docs.python.org/3/library/string.html#format-specification-mini-language)，以及强大的字符串插值。这些特性允许您添加变量，甚至是有效的 Python 表达式，并在将它们添加到[字符串](https://realpython.com/python-strings/)之前，在运行时对它们进行评估:

>>>

```py
>>> def get_name_and_decades(name, age):
...     return f"My name is {name} and I'm {age / 10:.5f} decades old."
...
>>> get_name_and_decades("Maria", 31)
My name is Maria and I'm 3.10000 decades old.
```

f-string 允许您将`Maria`放入字符串中，并在一个简洁的操作中添加她的年龄和所需的格式。

需要注意的一个风险是，如果您输出用户生成的值，那么这可能会引入安全风险，在这种情况下，[模板字符串](https://realpython.com/python-string-formatting/#4-template-strings-standard-library)可能是更安全的选择。

### 用`sorted()` 对复杂列表进行排序

大量的编码面试问题需要某种排序，有多种有效的方法可以对项目进行排序。除非面试官希望你实现自己的[排序算法](https://realpython.com/sorting-algorithms-python/)，通常最好用 [`sorted()`](https://realpython.com/python-sort/) 。

你可能见过排序的最简单的用法，比如对数字[或者字符串按照升序或者降序排序:](https://realpython.com/python-numbers/)

>>>

```py
>>> sorted([6,5,3,7,2,4,1])
[1, 2, 3, 4, 5, 6, 7]

>>> sorted(['cat', 'dog', 'cheetah', 'rhino', 'bear'], reverse=True)
['rhino', 'dog', 'cheetah', 'cat', 'bear]
```

默认情况下，`sorted()`已经按升序对输入进行了排序，而`reverse`关键字参数使它按降序排序。

值得一提的是可选的关键字参数`key`，它允许您指定一个函数，在排序之前对每个元素调用这个函数。添加函数允许自定义排序规则，这在您想要对更复杂的数据类型进行排序时尤其有用:

>>>

```py
>>> animals = [
...     {'type': 'penguin', 'name': 'Stephanie', 'age': 8},
...     {'type': 'elephant', 'name': 'Devon', 'age': 3},
...     {'type': 'puma', 'name': 'Moe', 'age': 5},
... ]
>>> sorted(animals, key=lambda animal: animal['age'])
[
 {'type': 'elephant', 'name': 'Devon', 'age': 3},
 {'type': 'puma', 'name': 'Moe', 'age': 5},
 {'type': 'penguin', 'name': 'Stephanie, 'age': 8},
]
```

通过传入一个返回每个元素年龄的 [lambda 函数](https://realpython.com/python-lambda/),您可以很容易地根据每个字典的单个值对字典列表进行排序。在这种情况下，字典现在按年龄升序排序。

[*Remove ads*](/account/join/)

## 有效利用数据结构

算法在编码面试中得到很多关注，但数据结构可能更重要。在编码面试环境中，选择正确的数据结构会对性能产生重大影响。

除了理论上的数据结构，Python 在其标准数据结构实现中内置了强大而方便的功能。这些数据结构在编写采访代码时非常有用，因为它们默认为您提供了许多功能，让您可以将时间集中在问题的其他部分。

### 用集合存储唯一值

您通常需要从现有数据集中移除重复的元素。新开发人员有时会在应该使用集合的时候使用列表，集合强制所有元素的唯一性。

假设你有一个名为`get_random_word()`的函数。它总是从一小组单词中随机选择:

>>>

```py
>>> import random
>>> all_words = "all the words in the world".split()
>>> def get_random_word():
...    return random.choice(all_words)
```

你应该反复调用`get_random_word()`来获得 1000 个随机单词，然后返回一个包含每个唯一单词的数据结构。这里有两种常见的次优方法和一种好方法。

**错误的方法**

`get_unique_words()`将值存储在列表中，然后将列表转换为集合:

>>>

```py
>>> def get_unique_words():
...     words = []
...     for _ in range(1000):
...         words.append(get_random_word())
...     return set(words)
>>> get_unique_words()
{'world', 'all', 'the', 'words'}
```

这种方法并不可怕，但是它不必要地创建了一个列表，然后将它转换成一个集合。面试官几乎总是注意到(并询问)这种类型的设计选择。

**更糟糕的方法**

为了避免从列表转换到集合，您现在将值存储在列表中，而不使用任何其他数据结构。然后，通过将新值与列表中当前的所有元素进行比较来测试唯一性:

>>>

```py
>>> def get_unique_words():
...     words = []
...     for _ in range(1000):
...         word = get_random_word()
...         if word not in words:
...             words.append(word)
...     return words
>>> get_unique_words()
['world', 'all', 'the', 'words']
```

这比第一种方法更糟糕，因为您必须将每个新单词与列表中已经存在的每个单词进行比较。这意味着随着单词数量的增长，查找的次数以二次方增长。换句话说，时间复杂度以 O(N)的数量级增长。

**好方法**

现在，您完全跳过使用列表，而是从一开始就使用集合:

>>>

```py
>>> def get_unique_words():
...     words = set()
...     for _ in range(1000):
...         words.add(get_random_word())
...     return words
>>> get_unique_words()
{'world', 'all', 'the', 'words'}
```

除了从一开始就使用集合之外，这看起来与其他方法没有太大的不同。如果您考虑在`.add()`中发生的事情，它甚至听起来像第二种方法:获取单词，检查它是否已经在集合中，如果不是，将它添加到数据结构中。

那么，为什么使用集合不同于第二种方法呢？

这是不同的，因为集合存储元素的方式允许以接近常数的时间检查一个值是否在集合中，不像列表需要线性时间查找。查找时间的差异意味着添加到集合的时间复杂度以 O(N)的速率增长，这在大多数情况下比第二种方法的 O(N)好得多。

[*Remove ads*](/account/join/)

### 使用发电机节省内存

列表理解是方便的工具，但有时会导致不必要的内存使用。

假设你被要求找出前 1000 个完美平方的总和，从 1 开始。您了解列表理解，因此您很快编写了一个可行的解决方案:

>>>

```py
>>> sum([i * i for i in range(1, 1001)])
333833500
```

您的解决方案列出了 1 到 1，000，000 之间的所有完美正方形，并将这些值相加。你的代码返回了正确的答案，但是你的面试官开始增加你需要求和的完美正方形的数量。

起初，你的函数不断弹出正确的答案，但很快它就开始变慢，直到最终这个过程似乎永远停止。这是你在编码面试中最不希望发生的事情。

这是怎么回事？

它会列出你要求的所有完美的正方形，然后把它们加起来。一个包含 1000 个完美方块的列表对计算机来说可能不算大，但 1 亿或 10 亿是相当多的信息，可以很容易地淹没计算机的可用内存资源。这就是这里正在发生的事情。

谢天谢地，有一个快速解决内存问题的方法。您只需用圆括号替换括号:

>>>

```py
>>> sum((i * i for i in range(1, 1001)))
333833500
```

交换括号会将你对列表的理解变成一个生成器表达式。当您知道要从序列中检索数据，但不需要同时访问所有数据时，生成器表达式是最理想的选择。

[生成器表达式](https://realpython.com/courses/python-generators/)返回一个`generator`对象，而不是创建一个列表。该对象知道自己在当前状态中的位置(例如，`i = 49`)，并且只在需要时才计算下一个值。

所以当`sum`通过反复调用`.__next__()`来迭代生成器对象时，生成器检查`i`等于多少，计算`i * i`，在内部递增`i`，并将适当的值返回给`sum`。该设计允许生成器用于大规模数据序列，因为一次只有一个元素存在于内存中。

### 用`.get()`和`.setdefault()` 定义字典中的默认值

最常见的编程任务之一是添加、修改或检索一个可能在字典中也可能不在字典中的条目。Python 字典具有优雅的功能，可以使这些任务变得简单明了，但是开发人员经常在不必要的时候显式地检查值。

假设你有一本名为`cowboy`的词典，你想得到那个牛仔的名字。一种方法是使用条件显式检查键:

>>>

```py
>>> cowboy = {'age': 32, 'horse': 'mustang', 'hat_size': 'large'}
>>> if 'name' in cowboy:
...     name = cowboy['name']
... else:
...     name = 'The Man with No Name'
...
>>> name
'The Man with No Name'
```

这种方法首先检查字典中是否存在`name`键，如果存在，则返回相应的值。否则，它将返回默认值。

虽然显式检查键确实有效，但是如果使用`.get()`，可以很容易地用一行代码替换它:

>>>

```py
>>> name = cowboy.get('name', 'The Man with No Name')
```

执行与第一种方法相同的操作，但现在它们是自动处理的。如果键存在，那么将返回正确的值。否则，将返回默认值。

但是，如果您想在访问`name`键的同时用默认值更新字典，该怎么办呢？`.get()`在这里并不能真正帮助您，所以您只能再次显式地检查值:

>>>

```py
>>> if 'name' not in cowboy:
...     cowboy['name'] = 'The Man with No Name'
...
>>> name = cowboy['name']
```

检查值并设置默认值是一种有效的方法，并且易于阅读，但是 Python 同样提供了一种更优雅的方法，使用`.setdefault()`:

>>>

```py
>>> name = cowboy.setdefault('name', 'The Man with No Name')
```

完成与上面的代码片段完全相同的事情。它检查`name`是否存在于`cowboy`中，如果存在，它返回该值。否则，它将`cowboy['name']`设置为`The Man with No Name`，并返回新值。

[*Remove ads*](/account/join/)

## 利用 Python 的标准库

默认情况下，Python 附带了许多功能，只需要一个`import`语句。它本身就很强大，但是知道如何利用标准库可以增强你的编码面试技巧。

很难从所有可用的模块中挑选出最有用的部分，因此本节将只关注其实用函数的一小部分。希望这些能对你编写面试代码有所帮助，并激发你学习更多关于这些和其他模块的高级功能的欲望。

### 用`collections.defaultdict()` 处理缺失的字典键

当您为单个键设置默认值时，`.get()`和`.setdefault()`工作得很好，但是通常需要为所有可能的未设置键设置默认值，特别是在编码面试环境中编程时。

假设你有一群学生，你需要记录他们的家庭作业成绩。输入值是一个格式为`(student_name, grade)`的元组列表，但是您想要轻松地查找单个学生的所有成绩，而不需要遍历列表。

存储成绩数据的一种方法是使用一个将学生姓名映射到成绩列表的字典:

>>>

```py
>>> student_grades = {}
>>> grades = [
...     ('elliot', 91),
...     ('neelam', 98),
...     ('bianca', 81),
...     ('elliot', 88),
... ]
>>> for name, grade in grades:
...     if name not in student_grades:
...         student_grades[name] = []
...     student_grades[name].append(grade)
...
>>> student_grades
{'elliot': [91, 88], 'neelam': [98], 'bianca': [81]}
```

在这种方法中，迭代学生并检查他们的名字是否已经是字典中的属性。如果没有，您可以将它们添加到字典中，并将空列表作为默认值。然后[将他们的实际成绩添加到学生的成绩列表中。](https://realpython.com/python-append/)

但是还有一种更简洁的方法，使用了一个`defaultdict`，它扩展了标准的`dict`功能，允许您设置一个缺省值，如果键不存在，将对该值进行操作:

>>>

```py
>>> from collections import defaultdict
>>> student_grades = defaultdict(list)
>>> for name, grade in grades:
...     student_grades[name].append(grade)
```

在这种情况下，您正在创建一个使用不带参数的`list()`构造函数作为默认工厂方法的`defaultdict`。没有参数的`list()`返回一个空列表，所以如果名字不存在的话`defaultdict`调用`list()`，然后允许附加等级。如果你想变得有趣，你也可以使用一个 lambda 函数作为你的工厂值来返回一个任意的常量。

利用`defaultdict`可以使应用程序代码更加整洁，因为您不必担心键级的默认值。相反，您可以在`defaultdict`级别处理它们一次，然后表现得好像密钥总是存在一样。有关这种技术的更多信息，请查看使用 Python defaultdict 类型处理丢失键的[。](https://realpython.com/python-defaultdict/)

### 用`collections.Counter` 计数可散列对象

您有一长串没有标点符号或大写字母的单词，并且您想要计算每个单词出现的次数。

您可以使用字典或`defaultdict`来增加计数，但是`collections.Counter`提供了一种更干净、更方便的方式来实现这一点。Counter 是`dict`的一个子类，它使用`0`作为任何缺失元素的默认值，并使计算对象的出现次数变得更容易:

>>>

```py
>>> from collections import Counter
>>> words = "if there was there was but if \
... there was not there was not".split()
>>> counts = Counter(words)
>>> counts
Counter({'if': 2, 'there': 4, 'was': 4, 'not': 2, 'but': 1})
```

当您将单词列表传递给`Counter`时，它会存储每个单词以及该单词在列表中出现的次数。

你好奇最常见的两个词是什么吗？只需使用`.most_common()`:

>>>

```py
>>> counts.most_common(2)
[('there', 4), ('was', 4)]
```

`.most_common()`是一个方便的方法，简单地通过计数返回最频繁的输入`n`。

[*Remove ads*](/account/join/)

### 使用`string`常量访问公共字符串组

现在是问答时间！`'A' > 'a'`是真还是假？

是假的，因为`A`的 ASCII 码是 65，但是`a`是 97，65 不大于 97。

为什么答案很重要？因为如果你想检查一个字符是否是英语字母表的一部分，一个流行的方法是看它是否在`A`和`z`之间(ASCII 表上的 65 和 122)。

检查 ASCII 代码是可行的，但在编码面试中很笨拙，很容易搞砸，特别是如果你不记得是小写还是大写的 ASCII 字符先出现。使用定义为 [`string`模块](https://docs.python.org/3/library/string.html)一部分的常量要容易得多。

您可以在`is_upper()`中看到一个正在使用的，它返回一个字符串中的所有字符是否都是大写字母:

>>>

```py
>>> import string
>>> def is_upper(word):
...     for letter in word:
...         if letter not in string.ascii_uppercase:
...             return False
...     return True
...
>>> is_upper('Thanks Geir')
False
>>> is_upper('LOL')
True
```

`is_upper()`遍历`word`中的字母，并检查这些字母是否是`string.ascii_uppercase`的一部分。如果你打印出`string.ascii_uppercase`，你会看到它只是一个低级的字符串。该值被设置为文字`'ABCDEFGHIJKLMNOPQRSTUVWXYZ'`。

所有的`string`常量都只是被频繁引用的字符串值的字符串。它们包括以下内容:

*   `string.ascii_letters`
*   `string.ascii_uppercase`
*   `string.ascii_lowercase`
*   `string.digits`
*   `string.hexdigits`
*   `string.octdigits`
*   `string.punctuation`
*   `string.printable`
*   `string.whitespace`

这些更容易使用，更重要的是，更容易阅读。

### 用`itertools` 生成排列组合

面试官喜欢给出真实的生活场景，让编码面试看起来不那么吓人，所以这里有一个人为的例子:你去一个游乐园，决定找出每一对可能一起坐在过山车上的朋友。

除非生成这些配对是面试问题的主要目的，否则生成所有可能的配对很可能只是通向工作算法的冗长乏味的一步。你可以用嵌套的 for 循环自己计算它们，或者你可以使用强大的 [`itertools`库](https://realpython.com/python-itertools/)。

`itertools`有多种工具可以生成可迭代的输入数据序列，但是现在我们只关注两个常见的函数:`itertools.permutations()`和`itertools.combinations()`。

`itertools.permutations()`构建所有排列的列表，这意味着它是长度与`count`参数匹配的输入值的每个可能分组的列表。`r`关键字参数让我们指定每个分组中有多少个值:

>>>

```py
>>> import itertools
>>> friends = ['Monique', 'Ashish', 'Devon', 'Bernie']
>>> list(itertools.permutations(friends, r=2))
[('Monique', 'Ashish'), ('Monique', 'Devon'), ('Monique', 'Bernie'),
('Ashish', 'Monique'), ('Ashish', 'Devon'), ('Ashish', 'Bernie'),
('Devon', 'Monique'), ('Devon', 'Ashish'), ('Devon', 'Bernie'),
('Bernie', 'Monique'), ('Bernie', 'Ashish'), ('Bernie', 'Devon')]
```

对于排列，元素的顺序很重要，所以`('sam', 'devon')`代表与`('devon', 'sam')`不同的配对，这意味着它们都将包含在列表中。

`itertools.combinations()`构建组合。这些也是输入值的可能分组，但现在值的顺序无关紧要了。因为`('sam', 'devon')`和`('devon', 'sam')`表示同一对，所以它们中只有一个会包含在输出列表中:

>>>

```py
>>> list(itertools.combinations(friends, r=2))
[('Monique', 'Ashish'), ('Monique', 'Devon'), ('Monique', 'Bernie'),
('Ashish', 'Devon'), ('Ashish', 'Bernie'), ('Devon', 'Bernie')]
```

因为值的顺序与组合无关，所以对于相同的输入列表，组合比排列要少。同样，因为我们将`r`设置为 2，所以每个分组中都有两个名字。

`.combinations()`和`.permutations()`只是一个强大的库的小例子，但是当你试图快速解决一个算法问题时，即使这两个函数也非常有用。

[*Remove ads*](/account/join/)

## 结论:编码面试超能力

在下一次编码面试中，您现在可以放心地使用 Python 的一些不太常见但更强大的标准特性了。关于这门语言整体上还有很多需要学习，但是这篇文章应该给你一个更深入的起点，同时让你在面试时更有效地使用 Python。

在本文中，您学习了不同类型的标准工具来增强您的编码面试技能:

*   [强大的内置功能](https://docs.python.org/3/library/functions.html#built-in-functions)
*   构建数据结构来处理普通场景，几乎不需要任何代码
*   针对特定问题的功能丰富的解决方案的标准库包，让您更快地编写更好的代码

面试可能不是真实软件开发的最佳近似，但了解如何在任何编程环境中取得成功是值得的，即使是面试。令人欣慰的是，在编码面试中学习如何使用 Python 可以帮助你更深入地理解这门语言，这将在日常开发中带来回报。

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**Python 编码面试:技巧&最佳实践**](/courses/python-coding-interviews-tips-best-practices/)*********