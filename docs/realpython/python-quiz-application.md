# 用 Python 构建一个测验应用程序

> 原文：<https://realpython.com/python-quiz-application/>

在本教程中，您将为终端构建一个 **Python 问答应用程序**。**竞猜**这个词是[在 1781 年首次使用](https://www.merriam-webster.com/dictionary/quiz#first-known)来表示*古怪的人*。如今，它主要用于描述一些琐事或专业知识的简短测试，问题如下:

> 单词*的第一次使用是在什么时候？*

通过遵循这个逐步的项目，您将构建一个可以测试一个人在一系列主题上的专业知识的应用程序。你可以用这个项目来强化你自己的知识或者挑战你的朋友来一场有趣的斗智。

**在本教程中，您将学习如何:**

*   **在终端与用户交互**
*   **提高**应用程序的可用性
*   **重构**你的应用程序，不断改进它
*   **将**数据存储在专用数据文件中

测验应用程序是一个综合性的项目，适合任何熟悉 Python 基础的人。在整个教程中，您将在单独的小步骤中获得所需的所有代码。您也可以通过点击下面的链接找到该应用程序的完整源代码:

**获取源代码:** [单击此处获取您将用于构建测验应用程序的源代码](https://realpython.com/bonus/python-quiz-application-project-code/)。

不管你是不是一个古怪的人，继续读下去，学习如何创建你自己的测验。

## 演示:您的 Python 测试应用程序

在这个循序渐进的项目中，您将构建一个终端应用程序，它可以就一系列主题对您和您的朋友进行测验:

[https://player.vimeo.com/video/717554866?background=1](https://player.vimeo.com/video/717554866?background=1)

你首先为你的问题选择一个主题。然后，对于每个问题，你将从一组选项中选择一个答案。有些问题可能有多个正确答案。你可以访问一个提示来帮助你。回答完一个问题后，你会读到一个解释，它可以为答案提供更多的背景信息。

[*Remove ads*](/account/join/)

## 项目概述

首先，您将创建一个基本的 Python 测验应用程序，它只能提问、收集答案并检查答案是否正确。从那里开始，你将添加越来越多的功能，以使你的应用程序更有趣，更友好，更有趣。

您将通过以下步骤迭代构建测验应用程序:

1.  创建一个可以提出多项选择问题的基本应用程序。
2.  通过改善应用程序的外观和处理用户错误的方式，使应用程序更加用户友好。
3.  重构代码以使用函数。
4.  通过将问题存储在专用数据文件中，将问题数据与源代码分开。
5.  扩展应用程序以处理多个正确答案，给出提示，并提供解释。
6.  支持不同的测验题目供选择，增加趣味性。

随着您的深入，您将获得从一个小脚本开始并扩展它的经验。这本身就是一项重要的技能。你最喜欢的程序、应用或游戏可能是从一个小的概念验证开始的，后来发展成今天的样子。

## 先决条件

在本教程中，您将使用 Python 的基本构件构建一个测验应用程序。在完成这些步骤时，如果您熟悉以下概念，将会很有帮助:

*   [读取终端用户的输入](https://realpython.com/python-input-output/)
*   在[结构](https://realpython.com/python-data-structures/)中组织数据，如[列表](https://realpython.com/python-lists-tuples/#python-lists)、[元组](https://realpython.com/python-lists-tuples/#python-tuples)和[字典](https://realpython.com/python-dicts/)
*   使用 [`if`语句](https://realpython.com/python-conditional-statements/)来检查不同的条件
*   用 [`for`](https://realpython.com/python-for-loop/) 和 [`while`](https://realpython.com/python-while-loop/) 循环重复动作
*   用[函数](https://realpython.com/defining-your-own-python-function/)封装代码

如果你对这些先决条件的知识没有信心，那也没关系！事实上，阅读本教程将有助于你实践这些概念。如果遇到困难，你可以随时停下来复习上面链接的资源。

## 第一步:提问

在这一步中，您将学习如何创建一个可以提问和检查答案的程序。这将是您的测验应用程序的基础，您将在本教程的剩余部分对其进行改进。在这一步结束时，您的程序将如下所示:

[https://player.vimeo.com/video/717554848?background=1](https://player.vimeo.com/video/717554848?background=1)

你的程序将能够提问和检查答案。这个版本包括您需要的基本功能，但是您将在后面的步骤中添加更多的功能。如果您愿意，那么您可以通过点击下面的链接并进入`source_code_step_1`目录来下载完成这一步后的源代码:

**获取源代码:** [单击此处获取您将用于构建测验应用程序的源代码](https://realpython.com/bonus/python-quiz-application-project-code/)。

### 用`input()` 获取用户信息

Python 的内置函数之一是 [`input()`](https://realpython.com/python-input-output/#reading-input-from-the-keyboard) 。您可以使用它从用户那里获取信息。对于第一个例子，在 Python [REPL](https://realpython.com/interacting-with-python/) 中运行以下代码:

>>>

```py
>>> name = input("What's your name? ")
What's your name? Geir Arne 
>>> name
'Geir Arne'
```

`input()`在用户输入信息前显示可选提示。在上面的例子中，提示显示在突出显示的行中，用户在点击 `Enter` 之前输入`Geir Arne`。无论用户输入什么，都会从`input()`返回。这在 REPL 的例子中可以看到，因为字符串`'Geir Arne'`被分配给了`name`。

你可以使用`input()`让 Python 向你提问并检查你的答案。尝试以下方法:

>>>

```py
>>> answer = input("When was the first known use of the word 'quiz'? ")
When was the first known use of the word 'quiz'? 1781 
>>> answer == 1781
False

>>> answer == "1781"
True
```

这个例子显示了您需要注意的一件事:`input()`总是返回一个文本字符串，即使该字符串只包含数字。您很快就会看到，这对于测验应用程序来说不是问题。然而，如果你想用`input()`的结果进行数学[计算](https://realpython.com/python-pyqt-gui-calculator/)，那么你需要先用[转换](https://realpython.com/python-dice-roll/#parse-and-validate-the-users-input)。

是时候开始构建您的测验应用程序了。打开编辑器，创建包含以下内容的文件`quiz.py`:

```py
# quiz.py

answer = input("When was the first known use of the word 'quiz'? ")
if answer == "1781":
    print("Correct!")
else:
    print(f"The answer is '1781', not {answer!r}")
```

这段代码与您在上面的 REPL 中所做的实验非常相似。您可以[运行](https://realpython.com/run-python-scripts/)您的应用程序来检查您的知识:

```py
$ python quiz.py
When was the first known use of the word 'quiz'? 1871 The answer is '1781', not '1871'
```

如果你碰巧给出了错误的答案，那么你会被温和地纠正，这样你下次就有希望做得更好。

**注意:**在`else`子句中，您引用的字符串文字前面的`f`表示该字符串是一个**格式的字符串**，通常称为 [f 字符串](https://realpython.com/python-f-strings/)。Python 对 f-strings 中花括号(`{}`)内的表达式求值，并将它们插入到字符串中。你可以选择添加不同的[格式说明符](https://realpython.com/python-formatted-output/#the-format_spec-component)。

例如，`!r` [指示](https://realpython.com/python-formatted-output/#the-conversion-component)应该根据其`repr()`表示插入`answer`。实际上，这意味着字符串用单引号括起来，就像`'1871'`。

只有一个问题的测验并不令人兴奋！您可以通过重复您的代码来提出另一个问题:

```py
# quiz.py

answer = input("When was the first known use of the word 'quiz'? ")
if answer == "1781":
    print("Correct!")
else:
    print(f"The answer is '1781', not {answer!r}")

answer = input("Which built-in function can get information from the user? ")
if answer == "input":
    print("Correct!")
else:
    print(f"The answer is 'input', not {answer!r}")
```

您通过复制和粘贴前面的代码添加了一个问题，然后更改了问题文本和正确答案。同样，您可以通过运行脚本来测试这一点:

```py
$ python quiz.py
When was the first known use of the word 'quiz'? 1781 Correct!
Which built-in function can get information from the user? get The answer is 'input', not 'get'
```

有用！然而，像这样复制和粘贴代码并不好。有一个[编程原则](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself)叫做**不要重复自己**(干)，它说你通常应该避免重复的代码，因为它变得难以维护。

接下来，您将开始改进您的代码，使其更容易使用。

[*Remove ads*](/account/join/)

### 使用列表和元组避免重复代码

Python 提供了几种灵活而强大的[数据结构](https://realpython.com/python-data-structures/)。通常可以用一个[元组](https://realpython.com/python-lists-tuples/#python-tuples)、一个[列表](https://realpython.com/python-lists-tuples/#python-lists)或者一个[字典](https://realpython.com/python-dicts/)结合一个 [`for`](https://realpython.com/python-for-loop/) 循环或者一个 [`while`](https://realpython.com/python-while-loop/) 循环来替换重复的代码。

代替重复代码，您将把您的问题和答案视为数据，并将它们移动到您的代码可以循环的数据结构中。接下来，迫在眉睫且通常具有挑战性的问题变成了应该如何组织数据。

从来没有唯一完美的数据结构。你通常会在几个选项中做出选择。在本教程中，随着应用程序的增长，您将多次重新考虑数据结构的选择。

现在，选择一个相当简单的数据结构:

*   一个列表将包含几个问题元素。
*   每个问题元素都是由问题文本和答案组成的二元组。

然后，您可以按如下方式存储您的问题:

```py
[
    ("When was the first known use of the word 'quiz'", "1781"),
    ("Which built-in function can get information from the user", "input"),
]
```

这非常符合您希望如何使用您的数据。您将循环每个问题，对于每个问题，您都希望访问问题和答案。

更改您的`quiz.py`文件，以便将您的问题和答案存储在`QUESTIONS`数据结构中:

```py
# quiz.py

QUESTIONS = [
    ("When was the first known use of the word 'quiz'", "1781"),
    ("Which built-in function can get information from the user", "input"),
    ("Which keyword do you use to loop over a given list of elements", "for")
]

for question, correct_answer in QUESTIONS:
    answer = input(f"{question}? ")
    if answer == correct_answer:
        print("Correct!")
    else:
        print(f"The answer is {correct_answer!r}, not {answer!r}")
```

当您运行这段代码时，它看起来与之前没有任何不同。事实上，您没有添加任何新功能。相反，你已经[重构了](https://realpython.com/python-refactoring/)你的代码，这样就可以更容易地向你的应用程序添加更多的问题。

在先前版本的代码中，您需要为添加的每个问题添加五行新代码。现在，`for`循环负责为每个问题运行这五行。要添加一个新问题，您只需要添加一行拼写出问题和相应的答案。

注意:在本教程中，你将学习一些小测验，所以问题和答案很重要。每个代码示例都会引入一个新问题。为了将本教程中的代码清单保持在可管理的大小，一些旧的问题可能会被删除。但是，请随意在代码中保留所有问题，或者甚至用您自己的问题和答案来替换它们。

你将在示例中看到的问题与教程相关，即使你不会在文本中找到所有的答案。如果你对某个问题或某个答案的更多细节感到好奇，请随意在网上搜索。

接下来，您将通过为每个问题添加备选答案来使您的测验应用程序更易于使用。

### 提供多种选择

使用`input()`是读取用户输入的一个好方法。然而，你目前使用它的方式可能会令人沮丧。例如，有人可能会这样回答你的一个问题:

```py
Which built-in function can get information from the user? input() The answer is 'input', not 'input()'
```

它们真的应该被标记为错误的吗？因为它们包含了括号来表示函数是可调用的。通过给用户提供替代方案，你可以为他们省去许多猜测。例如:

```py
 - get
 - input
 - print
 - write
Which built-in function can get information from the user? input Correct!
```

这里，备选项表明您希望输入不带括号的答案。在本例中，选项列在问题之前。这有点违反直觉，但更容易在您当前的代码中实现。您将在下一步的[中对此进行改进。](#step-2-make-your-application-user-friendly)

为了实现备选答案，您需要您的数据结构能够记录每个问题的三条信息:

1.  问题文本
2.  正确答案
3.  回答备选方案

是时候第一次——但不是最后一次——重访`QUESTIONS`,并对它做些改变了。将备选答案存储在列表中是有意义的，因为可以有任意数量的备选答案，而您只想将它们显示在屏幕上。此外，您可以将正确的答案视为备选答案之一，并将其包含在列表中，只要您以后能够检索到它。

你决定将`QUESTIONS`改成一本字典，其中的关键字是你的问题，值是备选答案列表。你总是把正确的答案放在选项列表的第一项，这样你就能识别它。

**注意:**您可以继续使用二元组列表来保存您的问题。事实上，您只是在迭代问题和答案，而不是通过使用问题作为关键字来查找答案。因此，您可能会认为元组列表比字典更适合您的用例。

但是，您使用字典是因为它在您的代码中看起来更好，并且问题和答案选项的角色更明显。

您更新了代码，以循环遍历新生成的字典中的每个条目。对于每个问题，您从选项中选出正确答案，并在提问前打印出所有选项:

```py
# quiz.py

QUESTIONS = {
    "When was the first known use of the word 'quiz'": [
        "1781", "1771", "1871", "1881"
    ],
    "Which built-in function can get information from the user": [
        "input", "get", "print", "write"
    ],
    "Which keyword do you use to loop over a given list of elements": [
        "for", "while", "each", "loop"
    ],
    "What's the purpose of the built-in zip() function": [
        "To iterate over two or more sequences at the same time",
        "To combine several strings into one",
        "To compress several files into one archive",
        "To get information from the user",
    ],
}

for question, alternatives in QUESTIONS.items():
    correct_answer = alternatives[0]
    for alternative in sorted(alternatives):
        print(f"  - {alternative}")

    answer = input(f"{question}? ")
    if answer == correct_answer:
        print("Correct!")
    else:
        print(f"The answer is {correct_answer!r}, not {answer!r}")
```

如果你总是把正确的答案作为第一选择，那么你的用户很快就会明白，并且每次都能猜出正确的答案。相反，你可以通过对选项进行排序来改变它们的顺序。测试您的应用:

```py
$ python quiz.py
 - 1771
 - 1781
 - 1871
 - 1881
When was the first known use of the word 'quiz'? 1781 Correct!

...

 - To combine several strings into one
 - To compress several files into one archive
 - To get information from the user
 - To iterate over two or more sequences at the same time
What's the purpose of the built-in zip() function?
 To itertate over two or more sequences at the same time The answer is 'To iterate over two or more sequences at the same time',
 not 'To itertate over two or more sequences at the same time'
```

最后一个问题揭示了另一个让用户感到沮丧的体验。在这个例子中，他们选择了正确的选项。然而，当他们打字的时候，一个打字错误溜了进来。你能让你的应用程序更宽容吗？

你知道用户会用其中一个选项来回答，所以你只需要一种方式让他们交流他们选择了哪个选项。您可以为每个备选项添加一个标签，并且只要求用户输入标签。

更新应用程序，使用 [`enumerate()`](https://realpython.com/python-enumerate/) 打印每个备选答案的索引:

```py
# quiz.py

QUESTIONS = {
    "Which keyword do you use to loop over a given list of elements": [
        "for", "while", "each", "loop"
    ],
    "What's the purpose of the built-in zip() function": [
        "To iterate over two or more sequences at the same time",
        "To combine several strings into one",
        "To compress several files into one archive",
        "To get information from the user",
    ],
    "What's the name of Python's sorting algorithm": [
        "Timsort", "Quicksort", "Merge sort", "Bubble sort"
    ],
}

for question, alternatives in QUESTIONS.items():
    correct_answer = alternatives[0]
 sorted_alternatives = sorted(alternatives) for label, alternative in enumerate(sorted_alternatives): print(f" {label}) {alternative}") 
 answer_label = int(input(f"{question}? ")) answer = sorted_alternatives[answer_label]    if answer == correct_answer:
        print("Correct!")
    else:
        print(f"The answer is {correct_answer!r}, not {answer!r}")
```

您将重新排序的选项存储为`sorted_alternatives`，这样您就可以根据用户输入的答案标签来查找完整的答案。回想一下，`input()`总是返回一个字符串，所以在将它作为一个列表索引之前，您需要用[将它转换成一个整数。](https://realpython.com/convert-python-string-to-int/)

现在，回答问题更方便了:

```py
$ python quiz.py
 0) each
 1) for
 2) loop
 3) while
Which keyword do you use to loop over a given list of elements? 2 The answer is 'for', not 'loop'
 0) To combine several strings into one
 1) To compress several files into one archive
 2) To get information from the user
 3) To iterate over two or more sequences at the same time
What's the purpose of the built-in zip() function? 3 Correct!
 0) Bubble sort
 1) Merge sort
 2) Quicksort
 3) Timsort
What's the name of Python's sorting algorithm? 3 Correct!
```

太好了！您已经创建了一个相当有能力的测验应用程序！在下一步中，您不会添加更多的功能。相反，您将使您的应用程序更加用户友好。

[*Remove ads*](/account/join/)

## 步骤 2:让你的应用程序对用户友好

在第二步中，您将改进测验应用程序，使其更易于使用。特别是，您将改进以下内容:

*   应用程序的外观和感觉
*   你如何总结用户的结果
*   如果你的用户输入了一个不存在的选项，会发生什么
*   你以什么顺序提出问题和选择

在这一步结束时，您的应用程序将如下工作:

[https://player.vimeo.com/video/717554822?background=1](https://player.vimeo.com/video/717554822?background=1)

你的程序仍然会像现在一样工作，但是它会更健壮，更有吸引力。您可以通过点击下面的链接在`source_code_step_2`目录中找到源代码，因为它将在本步骤结束时出现:

**获取源代码:** [单击此处获取您将用于构建测验应用程序的源代码](https://realpython.com/bonus/python-quiz-application-project-code/)。

### 更好地格式化输出

[回顾一下](#provide-multiple-choices)你的测验申请目前是如何呈现的。不是很吸引人。没有空行告诉你新问题从哪里开始，备选项列在问题上面，有点混乱。此外，不同选项的编号从`0`开始，而不是从`1`开始，这将更加自然。

在下一次对`quiz.py`的更新中，你将对问题本身进行编号，并在备选答案上方显示问题文本。此外，您将使用小写字母而不是数字来标识答案:

```py
# quiz.py

from string import ascii_lowercase 
QUESTIONS = {
    "What's the purpose of the built-in zip() function": [
        "To iterate over two or more sequences at the same time",
        "To combine several strings into one",
        "To compress several files into one archive",
        "To get information from the user",
    ],
    "What's the name of Python's sorting algorithm": [
        "Timsort", "Quicksort", "Merge sort", "Bubble sort"
    ],
    "What does dict.get(key) return if key isn't found in dict": [
        "None", "key", "True", "False",
    ]
}

for num, (question, alternatives) in enumerate(QUESTIONS.items(), start=1):
 print(f"\nQuestion {num}:") print(f"{question}?")    correct_answer = alternatives[0]
 labeled_alternatives = dict(zip(ascii_lowercase, sorted(alternatives))) for label, alternative in labeled_alternatives.items():        print(f" {label}) {alternative}")

 answer_label = input("\nChoice? ") answer = labeled_alternatives.get(answer_label)    if answer == correct_answer:
 print("⭐ Correct! ⭐")    else:
        print(f"The answer is {correct_answer!r}, not {answer!r}")
```

您使用`string.ascii_lowercase`来获得标记您的备选答案的字母。您用`zip()`组合字母和备选词，并将其存储在字典中，如下所示:

>>>

```py
>>> import string
>>> dict(zip(string.ascii_lowercase, ["1771", "1781", "1871", "1881"]))
{'a': '1771', 'b': '1781', 'c': '1871', 'd': '1881'}
```

当您向用户显示选项时，以及当您根据用户输入的标签查找用户的答案时，可以使用这些带标签的备选项。注意特殊转义字符串`"\n"`的使用。这被解释为换行并在屏幕上添加一个空行。这是向输出添加一些组织的简单方法:

```py
$ python quiz.py

Question 1:
What's the purpose of the built-in zip() function?
 a) To combine several strings into one
 b) To compress several files into one archive
 c) To get information from the user
 d) To iterate over two or more sequences at the same time

Choice? d ⭐ Correct! ⭐

Question 2:
What's the name of Python's sorting algorithm?
 a) Bubble sort
 b) Merge sort
 c) Quicksort
 d) Timsort

Choice? c The answer is 'Timsort', not 'Quicksort'
```

在终端中，您的输出仍然大多是单色的，但它在视觉上更令人愉悦，也更容易阅读。

### 保持分数

既然您已经对问题进行了编号，那么跟踪用户正确回答了多少问题也是很好的。您可以添加一个变量`num_correct`来处理这个问题:

```py
# quiz.py

from string import ascii_lowercase

QUESTIONS = {
    "What does dict.get(key) return if key isn't found in dict": [
        "None", "key", "True", "False",
    ],
    "How do you iterate over both indices and elements in an iterable": [
        "enumerate(iterable)",
        "enumerate(iterable, start=1)",
        "range(iterable)",
        "range(iterable, start=1)",
    ],
}

num_correct = 0 for num, (question, alternatives) in enumerate(QUESTIONS.items(), start=1):
    print(f"\nQuestion {num}:")
    print(f"{question}?")
    correct_answer = alternatives[0]
    labeled_alternatives = dict(zip(ascii_lowercase, sorted(alternatives)))
    for label, alternative in labeled_alternatives.items():
        print(f" {label}) {alternative}")

    answer_label = input("\nChoice? ")
    answer = labeled_alternatives.get(answer_label)
    if answer == correct_answer:
 num_correct += 1        print("⭐ Correct! ⭐")
    else:
        print(f"The answer is {correct_answer!r}, not {answer!r}")

print(f"\nYou got {num_correct} correct out of {num} questions")
```

你每答对一个，就增加`num_correct`。`num`循环变量已经计算了问题的总数，因此您可以使用它来报告用户的结果。

[*Remove ads*](/account/join/)

### 处理用户错误

到目前为止，您还没有太担心如果用户输入无效的答案会发生什么。在不同版本的应用程序中，这种疏忽可能会导致程序产生一个[错误](https://realpython.com/python-traceback/)，或者——不太明显——将用户的无效答案注册为错误。

当用户输入无效内容时，允许用户重新输入他们的答案，可以更好地处理用户错误。一种方法是将`input()`包装在一个`while`循环中:

>>>

```py
>>> while (text := input()) != "quit":
...     print(f"Echo: {text}")
...
Hello! Echo: Hello!
Walrus ... Echo: Walrus ...
quit
```

条件`(text := input()) != "quit"`同时做几件事。它使用一个赋值表达式(`:=`)，通常称为 [walrus 操作符](https://realpython.com/python-walrus-operator/)，将用户输入存储为`text`，并将其与字符串`"quit"`进行比较。while 循环将一直运行，直到您在提示符下键入`quit`。更多例子见[海象操作符:Python 3.8 赋值表达式](https://realpython.com/python-walrus-operator/#while-loops)。

**注意:**如果你使用的是比 3.8 更老的 Python 版本，那么赋值表达式将导致一个[语法错误](https://realpython.com/invalid-syntax-python/)。你可以[重写](https://realpython.com/python-walrus-operator/#while-loops)代码来避免使用 walrus 操作符。在您之前下载的源代码中有一个运行在 Python 3.7 上的测验应用程序版本。

在您的测验应用程序中，您使用类似的构造进行循环，直到用户给出有效答案:

```py
# quiz.py

from string import ascii_lowercase

QUESTIONS = {
    "How do you iterate over both indices and elements in an iterable": [
        "enumerate(iterable)",
        "enumerate(iterable, start=1)",
        "range(iterable)",
        "range(iterable, start=1)",
    ],
    "What's the official name of the := operator": [
        "Assignment expression",
        "Named expression",
        "Walrus operator",
        "Colon equals operator",
    ],
}

num_correct = 0
for num, (question, alternatives) in enumerate(QUESTIONS.items(), start=1):
    print(f"\nQuestion {num}:")
    print(f"{question}?")
    correct_answer = alternatives[0]
    labeled_alternatives = dict(zip(ascii_lowercase, sorted(alternatives)))
    for label, alternative in labeled_alternatives.items():
        print(f" {label}) {alternative}")

 while (answer_label := input("\nChoice? ")) not in labeled_alternatives: print(f"Please answer one of {', '.join(labeled_alternatives)}") 
 answer = labeled_alternatives[answer_label]    if answer == correct_answer:
        num_correct += 1
        print("⭐ Correct! ⭐")
    else:
        print(f"The answer is {correct_answer!r}, not {answer!r}")

print(f"\nYou got {num_correct} correct out of {num} questions")
```

如果您在提示符下输入了一个无效的选项，那么系统会提醒您有效的选项:

```py
$ python quiz.py

Question 1:
How do you iterate over both indices and elements in an iterable?
 a) enumerate(iterable)
 b) enumerate(iterable, start=1)
 c) range(iterable)
 d) range(iterable, start=1)

Choice? e Please answer one of a, b, c, d

Choice? a ⭐ Correct! ⭐
```

请注意，一旦`while`循环退出，就可以保证`answer_label`是`labeled_alternatives`中的一个键，所以直接查找`answer`是安全的。接下来，您将通过在测验中注入一些随机性来增加一项改进。

### 为您的测验增加多样性

目前，当您运行测验应用程序时，您总是按照问题在源代码中列出的顺序来提问。此外，给定问题的备选答案也有固定的顺序，从不改变。

你可以稍微改变一下，给你的测验增加一些变化。您可以随机化问题的顺序和每个问题的备选答案的顺序:

```py
# quiz.py

import random from string import ascii_lowercase

NUM_QUESTIONS_PER_QUIZ = 5 QUESTIONS = {
    "What's the official name of the := operator": [
        "Assignment expression",
        "Named expression",
        "Walrus operator",
        "Colon equals operator",
    ],
    "What's one effect of calling random.seed(42)": [
        "The random numbers are reproducible.",
        "The random numbers are more random.",
        "The computer clock is reset.",
        "The first random number is always 42.",
    ]
}

num_questions = min(NUM_QUESTIONS_PER_QUIZ, len(QUESTIONS)) questions = random.sample(list(QUESTIONS.items()), k=num_questions) 
num_correct = 0
for num, (question, alternatives) in enumerate(questions, start=1):
    print(f"\nQuestion {num}:")
    print(f"{question}?")
    correct_answer = alternatives[0]
 labeled_alternatives = dict( zip(ascii_lowercase, random.sample(alternatives, k=len(alternatives))) )    for label, alternative in labeled_alternatives.items():
        print(f" {label}) {alternative}")

    while (answer_label := input("\nChoice? ")) not in labeled_alternatives:
        print(f"Please answer one of {', '.join(labeled_alternatives)}")

    answer = labeled_alternatives[answer_label]
    if answer == correct_answer:
        num_correct += 1
        print("⭐ Correct! ⭐")
    else:
        print(f"The answer is {correct_answer!r}, not {answer!r}")

print(f"\nYou got {num_correct} correct out of {num} questions")
```

你使用 [`random.sample()`](https://realpython.com/python-random/#the-random-module) 来随机排列你的问题和答案选项的顺序。通常，`random.sample()`从一个集合中随机挑选几个样本。但是，如果您要求的样本数与序列中的项目数一样多，那么您实际上是在随机地对整个序列进行重新排序:

>>>

```py
>>> import random
>>> random.sample(["one", "two", "three"], k=3)
['two', 'three', 'one']
```

此外，您[将测验中的问题数量限制为`NUM_QUESTIONS_PER_QUIZ`，最初设置为五个。如果你在申请中包含了五个以上的问题，那么除了提问的顺序之外，这也增加了提问问题的多样性。](https://realpython.com/python-min-and-max/#clipping-values-to-the-edges-of-an-interval)

**注:**你也可以用 [`random.shuffle()`](https://docs.python.org/3/library/random.html#random.shuffle) 来洗牌你的问题和备选方案。不同之处在于`shuffle()`就地重新排序序列，这意味着它改变了底层的`QUESTIONS`数据结构。 [`sample()`](https://docs.python.org/3/library/random.html#random.sample) 创建新的问题和替代列表。

在您当前的代码中，使用`shuffle()`不会有问题，因为`QUESTIONS`会在您每次运行测验应用程序时重置。这可能会成为一个问题，例如，如果你实现了多次询问同一个问题的可能性。如果不改变或改变底层数据结构，您的代码通常更容易推理。

在这一步中，您已经改进了测验应用程序。现在是时候退一步考虑代码本身了。在下一节中，您将重新组织代码，以便保持它的模块化并为进一步的开发做好准备。

[*Remove ads*](/account/join/)

## 步骤 3:用函数组织你的代码

在这一步，你将**重构**你的代码。[重构](https://realpython.com/python-refactoring/)意味着你将改变你的代码，但是你的应用程序的行为和用户的体验将保持不变。这听起来可能不是很令人兴奋，但它最终会非常有用，因为好的重构会使维护和扩展代码更加方便。

**注意:**如果你想看两个真正的 Python 团队成员如何重构一些代码，那么看看[重构:准备你的代码以获得帮助](https://realpython.com/courses/refactoring-code-to-get-help/)。您还将学习如何提出清晰、简洁的编程问题。

目前，你的代码不是特别有条理。你所有的陈述都是相当低级的。您将定义[函数](https://realpython.com/defining-your-own-python-function/)来改进您的代码。它们的一些优点如下:

*   函数名为**更高层次的操作**，可以帮助你获得代码的概观。
*   功能可以被**重用**。

要查看代码重构后的样子，请点击下面的链接，查看`source_code_step_3`文件夹:

**获取源代码:** [单击此处获取您将用于构建测验应用程序的源代码](https://realpython.com/bonus/python-quiz-application-project-code/)。

### 准备数据

许多游戏和应用程序都遵循一个共同的生命周期:

1.  **预处理:**准备初始数据。
2.  **流程:**运行主循环。
3.  **后处理:**清理并关闭应用程序。

在您的测验应用程序中，您首先阅读可用的问题，然后询问每个问题，最后报告最终分数。如果你回头看看你当前的代码，你会在代码中看到这三个步骤。但是这个组织仍然隐藏在所有的细节中。

通过将主要功能封装在一个函数中，可以使其更加清晰。您还不需要更新您的`quiz.py`文件，但是请注意，您可以将前面的段落翻译成如下所示的代码:

```py
def run_quiz():
    # Preprocess
    questions = prepare_questions()

    # Process (main loop)
    num_correct = 0
    for question in questions:
        num_correct += ask_question(question)

    # Postprocess
    print(f"\nYou got {num_correct} correct")
```

这段代码不会像现在这样运行。函数`prepare_questions()`和`ask_question()`还没有定义，还缺少一些其他的细节。尽管如此，`run_quiz()`在高层次上封装了应用程序的功能。

像这样在一个高层次上写下你的应用程序流可以是一个很好的开始来发现哪些函数是你的代码中的自然构建块。在本节的其余部分，您将填写缺失的详细信息:

*   执行`prepare_questions()`。
*   执行`ask_question()`。
*   重访`run_quiz()`。

您现在将对您的测验应用程序的代码进行相当大的修改，因为您正在重构它以使用函数。在这样做之前，最好确保您可以恢复到当前状态，您知道这是可行的。如果您使用的是[版本控制系统](https://realpython.com/python-git-github-intro/)，那么您可以通过用不同的文件名保存您代码的副本或者通过提交来做到这一点。

一旦你安全地存储了你当前的代码，从一个新的`quiz.py`开始，它只包含你的导入和[全局变量](https://realpython.com/python-scope-legb-rule/#modules-the-global-scope)。您可以从以前的版本中复制这些内容:

```py
# quiz.py

import random
from string import ascii_lowercase

NUM_QUESTIONS_PER_QUIZ = 5
QUESTIONS = {
    "What's one effect of calling random.seed(42)": [
        "The random numbers are reproducible.",
        "The random numbers are more random.",
        "The computer clock is reset.",
        "The first random number is always 42.",
    ],
    "When does __name__ == '__main__' equal True in a Python file": [
        "When the file is run as a script",
        "When the file is imported as a module",
        "When the file has a valid name",
        "When the file only has one function",
    ]
}
```

记住你只是在重组你的代码。您没有添加新的功能，所以您不需要导入任何新的库。

接下来，您将实现必要的预处理。在这种情况下，这意味着您将准备好`QUESTIONS`数据结构，以便在主循环中使用。目前，您可能会限制问题的数量，并确保它们以随机顺序列出:

```py
# quiz.py

# ...

def prepare_questions(questions, num_questions):
    num_questions = min(num_questions, len(questions))
    return random.sample(list(questions.items()), k=num_questions)
```

注意，`prepare_questions()`处理一般的`questions`和`num_questions`参数。随后，您将传入特定的`QUESTIONS`和`NUM_QUESTIONS_PER_QUIZ`作为参数。这意味着`prepare_questions()`不依赖于你的全局变量。有了这种分离，您的函数就更通用了，并且您以后可以更容易地替换问题的来源。

[*Remove ads*](/account/join/)

### 提问

回头看看`run_quiz()`函数的草图，记住它包含了你的主循环。对于每个问题，您将调用`ask_question()`。您下一个任务是实现助手函数。

思考`ask_question()`需要做什么:

1.  从选项列表中选出正确答案
2.  打乱选择
3.  将问题打印到屏幕上
4.  将所有备选项打印到屏幕上
5.  从用户那里得到答案
6.  检查用户的答案是否有效
7.  检查用户回答是否正确
8.  如果答案正确，将`1`加到正确答案的计数中

在一个功能中有很多小事情要做，你可以考虑是否有进一步模块化的潜力。例如，上面列表中的第 3 到第 6 项都是关于与用户交互的，您可以将它们放入另一个助手功能中。

为了实现这种模块化，将下面的`get_answer()`助手函数添加到您的源代码中:

```py
# quiz.py

# ...

def get_answer(question, alternatives):
    print(f"{question}?")
    labeled_alternatives = dict(zip(ascii_lowercase, alternatives))
    for label, alternative in labeled_alternatives.items():
        print(f" {label}) {alternative}")

    while (answer_label := input("\nChoice? ")) not in labeled_alternatives:
        print(f"Please answer one of {', '.join(labeled_alternatives)}")

    return labeled_alternatives[answer_label]
```

该函数接受一个问题文本和一个备选项列表。然后，使用与前面相同的技术来标记替代项，并要求用户输入一个有效的标签。最后，你返回用户的答案。

使用`get_answer()`简化了`ask_question()`的实现，因为您不再需要处理用户交互。您可以执行如下操作:

```py
# quiz.py

# ...

def ask_question(question, alternatives):
    correct_answer = alternatives[0]
    ordered_alternatives = random.sample(alternatives, k=len(alternatives))

    answer = get_answer(question, ordered_alternatives)
    if answer == correct_answer:
        print("⭐ Correct! ⭐")
        return 1
    else:
        print(f"The answer is {correct_answer!r}, not {answer!r}")
        return 0
```

像前面一样，首先使用`random.shuffle()`对答案选项进行随机重新排序。接下来，您调用`get_answer()`，它处理从用户那里获得答案的所有细节。因此，你可以通过检查答案的正确性来结束`ask_question()`。注意，您返回了`1`或`0`，这向调用函数表明答案是否正确。

**注意:**你可以用[布尔值](https://realpython.com/python-boolean/)替换返回值。代替`1`，你可以返回`True`，代替`0`，你可以返回`False`。这是可行的，因为 Python 在计算中将布尔值视为整数:

>>>

```py
>>> True + True
2

>>> True * False
0
```

在某些情况下，当你使用`True`和`False`时，你的代码读起来更自然。在这种情况下，您正在计算正确答案，因此使用数字似乎更直观。

您现在已经准备好正确地实现`run_quiz()`。在实现`prepare_questions()`和`ask_question()`时，你学到的一件事是你需要传递哪些参数:

```py
# quiz.py

# ...

def run_quiz():
    questions = prepare_questions(
        QUESTIONS, num_questions=NUM_QUESTIONS_PER_QUIZ
    )

    num_correct = 0
    for num, (question, alternatives) in enumerate(questions, start=1):
        print(f"\nQuestion {num}:")
        num_correct += ask_question(question, alternatives)

    print(f"\nYou got {num_correct} correct out of {num} questions")
```

如前所述，您使用`enumerate()`来保存一个计数器，对您提出的问题进行计数。你可以根据`ask_question()`的返回值增加`num_correct`。观察`run_quiz()`是您唯一直接与`QUESTIONS`和`NUM_QUESTIONS_PER_QUIZ`交互的功能。

您的重构现在已经完成，除了一件事。如果你现在跑`quiz.py`，那就好像什么都没发生。事实上，Python 会读取你的全局变量并定义你的函数。但是，您没有调用任何这些函数。因此，您需要添加一个启动应用程序的函数调用:

```py
# quiz.py

# ...

if __name__ == "__main__":
    run_quiz()
```

你在`quiz.py`的末尾调用`run_quiz()`，在任何函数之外。用一个 [`if __name__ == "__main__"`](https://realpython.com/if-name-main-python/) 测试来保护这样一个对主函数的调用是一个很好的实践。这个[特殊咒语](https://realpython.com/python-import/#import-scripts-as-modules)是一个 Python 约定，意思是当你作为脚本运行`quiz.py`时会调用`run_quiz()`，但是当你作为模块导入`quiz`时不会调用。

就是这样！您已经将代码重构为几个函数。这将有助于您跟踪应用程序的功能。这在本教程中也很有用，因为您可以考虑更改单个函数，而不是更改整个脚本。

对于本教程的其余部分，您将看到您的完整代码列在如下所示的可折叠框中。展开这些以查看当前状态，并获得整个应用程序的概述:



下面列出了测验应用程序的完整源代码:

```py
# quiz.py

import random
from string import ascii_lowercase

NUM_QUESTIONS_PER_QUIZ = 5
QUESTIONS = {
    "When was the first known use of the word 'quiz'": [
        "1781", "1771", "1871", "1881",
    ],
    "Which built-in function can get information from the user": [
        "input", "get", "print", "write",
    ],
    "Which keyword do you use to loop over a given list of elements": [
        "for", "while", "each", "loop",
    ],
    "What's the purpose of the built-in zip() function": [
        "To iterate over two or more sequences at the same time",
        "To combine several strings into one",
        "To compress several files into one archive",
        "To get information from the user",
    ],
    "What's the name of Python's sorting algorithm": [
        "Timsort", "Quicksort", "Merge sort", "Bubble sort",
    ],
    "What does dict.get(key) return if key isn't found in dict": [
        "None", "key", "True", "False",
    ],
    "How do you iterate over both indices and elements in an iterable": [
        "enumerate(iterable)",
        "enumerate(iterable, start=1)",
        "range(iterable)",
        "range(iterable, start=1)",
    ],
    "What's the official name of the := operator": [
        "Assignment expression",
        "Named expression",
        "Walrus operator",
        "Colon equals operator",
    ],
    "What's one effect of calling random.seed(42)": [
        "The random numbers are reproducible.",
        "The random numbers are more random.",
        "The computer clock is reset.",
        "The first random number is always 42.",
    ],
    "When does __name__ == '__main__' equal True in a Python file": [
        "When the file is run as a script",
        "When the file is imported as a module",
        "When the file has a valid name",
        "When the file only has one function",
    ]
}

def run_quiz():
    questions = prepare_questions(
        QUESTIONS, num_questions=NUM_QUESTIONS_PER_QUIZ
    )

    num_correct = 0
    for num, (question, alternatives) in enumerate(questions, start=1):
        print(f"\nQuestion {num}:")
        num_correct += ask_question(question, alternatives)

    print(f"\nYou got {num_correct} correct out of {num} questions")

def prepare_questions(questions, num_questions):
    num_questions = min(num_questions, len(questions))
    return random.sample(list(questions.items()), k=num_questions)

def ask_question(question, alternatives):
    correct_answer = alternatives[0]
    ordered_alternatives = random.sample(alternatives, k=len(alternatives))

    answer = get_answer(question, ordered_alternatives)
    if answer == correct_answer:
        print("⭐ Correct! ⭐")
        return 1
    else:
        print(f"The answer is {correct_answer!r}, not {answer!r}")
        return 0

def get_answer(question, alternatives):
    print(f"{question}?")
    labeled_alternatives = dict(zip(ascii_lowercase, alternatives))
    for label, alternative in labeled_alternatives.items():
        print(f" {label}) {alternative}")

    while (answer_label := input("\nChoice? ")) not in labeled_alternatives:
        print(f"Please answer one of {', '.join(labeled_alternatives)}")

    return labeled_alternatives[answer_label]

if __name__ == "__main__":
    run_quiz()
```

使用`python quiz.py`运行您的应用程序。

通过这一步，您已经重构了代码，使其更便于使用。您将命令分成了组织良好的功能，您可以继续开发这些功能。下一步，您将利用这一点，改进将问题读入应用程序的方式。

[*Remove ads*](/account/join/)

## 步骤 4:将数据分离到自己的文件中

在这一步中，您将继续您的重构之旅。现在你的重点是如何向你的申请提出问题。

到目前为止，您已经将问题直接存储在源代码的`QUESTIONS`数据结构中。通常最好将数据与代码分开。这种分离可以使您的代码更具可读性，但更重要的是，如果数据没有隐藏在您的代码中，您可以利用为处理数据而设计的系统。

在本节中，您将学习如何将您的问题存储在一个根据 [TOML 标准](https://toml.io/)格式化的单独数据文件中。其他选项——你不会在本教程中涉及——是以不同的文件格式存储问题，如 [JSON](https://realpython.com/python-json/) 或 [YAML](https://realpython.com/python-yaml/) ，或者将它们存储在数据库中，或者是传统的[关系数据库](https://realpython.com/python-mysql/)或者是 [NoSQL](https://realpython.com/introduction-to-mongodb-and-python/) 数据库。

要查看在这一步中您将如何改进您的代码，请单击下面并转到`source_code_step_4`目录:

**获取源代码:** [单击此处获取您将用于构建测验应用程序的源代码](https://realpython.com/bonus/python-quiz-application-project-code/)。

### 将问题移至 TOML 文件

TOML 被标榜为“人类的一种配置文件格式”([来源](https://toml.io/))。它被设计成人类可读，计算机解析也不复杂。信息用键值对表示，可以映射到一个[散列表](https://realpython.com/python-hash-table/)数据结构，就像 Python 字典一样。

TOML 支持几种数据类型，包括字符串、整数、浮点数、布尔值和日期。此外，数据可以以数组和表的形式组织，分别类似于 Python 的列表和字典。TOML 在过去几年里越来越受欢迎，在格式规范的[版本 1.0.0](https://toml.io/en/v1.0.0) 于 2021 年 1 月[发布](https://github.com/toml-lang/toml/releases/tag/1.0.0)后，该格式已经稳定。

创建一个名为`questions.toml`的新文本文件，并添加以下内容:

```py
# questions.toml "When does __name__ == '__main__' equal True in a Python file"  =  [ "When the file is run as a script", "When the file is imported as a module", "When the file has a valid name", "When the file only has one function", ] "Which version of Python is the first with TOML support built in"  =  [ "3.11",  "3.9",  "3.10",  "3.12" ]
```

虽然 TOML 语法和 Python 语法之间存在差异，但是您可以识别出一些元素，例如使用引号(`"`)表示文本，使用方括号(`[]`)表示元素列表。

要在 Python 中处理 TOML 文件，您需要一个解析它们的库。在本教程中，您将使用 [`tomli`](https://pypi.org/project/tomli/) 。这将是您在这个项目中使用的唯一一个不属于 Python 标准库的包。

**注意:** TOML 支持是[在 Python 3.11 中加入了](https://peps.python.org/pep-0680/)到 Python 的标准库中。如果您已经在使用 Python 3.11，那么您可以跳过下面的说明来创建一个虚拟环境并安装`tomli`。相反，您可以通过用兼容的`tomllib`替换代码中提到的任何`tomli`来立即开始编码。

在本节的后面，您将学习如何编写可以使用`tomllib`的代码(如果可用的话),并在必要时回退到`tomli`。

在安装`tomli`之前，您应该创建并激活一个虚拟环境:

*   [*视窗*](#windows-1)
**   [**Linux + macOS**](#linux-macos-1)*

```py
PS> python -m venv venv
PS> venv\Scripts\activate
```

```py
$ python -m venv venv
$ source venv/bin/activate
```

然后可以用 [`pip`](https://realpython.com/what-is-pip/) 安装`tomli`:

*   [*视窗*](#windows-2)
**   [**Linux + macOS**](#linux-macos-2)*

```py
(venv) PS> python -m pip install tomli
```

```py
(venv) $ python -m pip install tomli
```

您可以通过解析之前创建的`questions.toml`来检查是否有可用的`tomli`。打开您的 Python REPL 并测试以下代码:

>>>

```py
>>> import tomli
>>> with open("questions.toml", mode="rb") as toml_file:
...     questions = tomli.load(toml_file)
...

>>> questions
{"When does __name__ == '__main__' equal True in a Python file":
 ['When the file is run as a script',
 'When the file is imported as a module',
 'When the file has a valid name',
 'When the file only has one function'],
 'Which version of Python is the first with TOML support built-in':
 ['3.11', '3.9', '3.10', '3.12']}
```

首先，注意到`questions`是一个常规的 Python 字典，它与您目前使用的`QUESTIONS`数据结构具有相同的形式。

您可以使用`tomli`以两种不同的方式解析 TOML 信息。在上面的例子中，您使用`tomli.load()`从一个打开的文件句柄中读取 TOML。或者，您可以使用`tomli.loads()`从文本字符串中读取 TOML。

**注意:**在将文件传递给`tomli.load()`之前，需要使用`mode="rb"`以[二进制模式](https://realpython.com/read-write-files-python/#buffered-binary-file-types)打开文件。这样`tomli`可以确保 TOML 文件的 [UTF-8 编码](https://realpython.com/python-encodings-guide/#enter-unicode)被正确处理。

如果你使用`tomli.loads()`，那么你传入的字符串将被解释为 UTF-8。

通过更新代码的序言，您可以将 TOML 文件集成到测验应用程序中，您可以在其中进行导入并定义全局变量:

```py
# quiz.py

# ...

import pathlib
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

NUM_QUESTIONS_PER_QUIZ = 5
QUESTIONS_PATH = pathlib.Path(__file__).parent / "questions.toml"
QUESTIONS = tomllib.loads(QUESTIONS_PATH.read_text())

# ...
```

您没有像前面那样做一个简单的`import tomli`，而是将您的导入包装在一个`try` … `except`语句中，该语句首先尝试导入`tomllib`。如果失败，那么你导入`tomli`，但将其重命名为`tomllib`。这样做的效果是，如果 Python 3.11 `tomllib`可用，您将使用它，如果不可用，[将退回到](https://github.com/hukkin/tomli#building-a-tomlitomllib-compatibility-layer)状态。

您正在使用 [`pathlib`](https://realpython.com/python-pathlib/) 来处理到`questions.toml`的路径。不是硬编码到`questions.toml`的路径，而是依赖特殊的 [`__file__`](https://docs.python.org/3/reference/datamodel.html?highlight=__file__) 变量。实际上，你说它和你的`quiz.py`文件位于同一个目录。

最后，使用`read_text()`将 TOML 文件作为文本字符串读取，然后使用`loads()`将该字符串解析到字典中。正如您在前面的示例中看到的，加载 TOML 文件会产生与您之前的问题相同的数据结构。一旦您对`quiz.py`做了更改，您的测验应用程序应该仍然运行，尽管问题是在 TOML 文件中定义的，而不是在您的源代码中。

继续向您的 TOML 文件添加几个问题，以确认它正在被使用。

[*Remove ads*](/account/join/)

### 增加数据格式的灵活性

您已经将问题数据从源代码中移出，并将其转换为专用的数据文件格式。与常规的 Python 字典相比，TOML 的一个优点是，您可以在保持数据可读性和可维护性的同时，为数据添加更多的结构。

TOML 的一个显著特征是**表**。这些是映射到 Python 中嵌套字典的命名部分。此外，您可以使用表的**数组**，它们由 Python 中的字典列表表示。

你可以利用这些来更明确地定义你的问题。考虑下面的 TOML 片段:

```py
[[questions]] question  =  "Which version of Python is the first with TOML support built in" answer  =  "3.11" alternatives  =  ["3.9",  "3.10",  "3.12"]
```

常规表格以类似`[questions]`的单括号线开始。您可以使用双括号来表示一个表格数组，如上所示。您可以用`tomli`解析 TOML:

>>>

```py
>>> toml = """
... [[questions]]
... question = "Which version of Python is the first with TOML support built in"
... answer = "3.11"
... alternatives = ["3.9", "3.10", "3.12"]
... """

>>> import tomli
>>> tomli.loads(toml)
{'questions': [
 {
 'question': 'Which version of Python is the first with TOML support built in',
 'answer': '3.11',
 'alternatives': ['3.9', '3.10', '3.12']
 }
]}
```

这导致了一个嵌套的数据结构，带有一个外部字典，其中的`questions`键指向一个字典列表。内部字典有`question`、`answer`和`alternatives`键。

这个结构比你到目前为止使用的要复杂一些。然而，它也更加明确，您不需要依赖于约定，例如代表正确答案的第一个答案选项。

现在，您将转换您的测验应用程序，以便它利用这个新的数据结构来回答您的问题。首先，在`questions.toml`中重新格式化你的问题。您应该将它们格式化如下:

```py
# questions.toml [[questions]] question  =  "Which version of Python is the first with TOML support built in" answer  =  "3.11" alternatives  =  ["3.9",  "3.10",  "3.12"] [[questions]] question  =  "What's the name of the list-like data structure in TOML" answer  =  "Array" alternatives  =  ["List",  "Sequence",  "Set"]
```

每个问题都存储在一个单独的`questions`表中，表中有问题文本、正确答案和备选答案的键值对。

原则上，要使用新格式，您需要对应用程序源代码进行两处修改:

1.  阅读内部`questions`列表中的问题。
2.  提问时，使用内部问题词典。

这些更改触及到您的主数据结构，因此它们需要在整个代码中进行一些小的代码更改。

首先，改变从 TOML 文件中读取问题的方式:

```py
# quiz.py

# ...

NUM_QUESTIONS_PER_QUIZ = 5
QUESTIONS_PATH = pathlib.Path(__file__).parent / "questions.toml"

def run_quiz():
 questions = prepare_questions( QUESTIONS_PATH, num_questions=NUM_QUESTIONS_PER_QUIZ ) 
    num_correct = 0
 for num, question in enumerate(questions, start=1):        print(f"\nQuestion {num}:")
 num_correct += ask_question(question) 
    print(f"\nYou got {num_correct} correct out of {num} questions")

def prepare_questions(path, num_questions):
 questions = tomllib.loads(path.read_text())["questions"]    num_questions = min(num_questions, len(questions))
    return random.sample(questions, k=num_questions)
```

您更改`prepare_questions()`来读取 TOML 文件并挑选出`questions`列表。此外，您可以简化`run_quiz()`中的主循环，因为关于一个问题的所有信息都包含在字典中。您不需要分别跟踪问题文本和备选方案。

后一点也需要对`ask_question()`进行一些修改:

```py
# quiz.py

# ...

def ask_question(question):
 correct_answer = question["answer"] alternatives = [question["answer"]] + question["alternatives"]    ordered_alternatives = random.sample(alternatives, k=len(alternatives))

 answer = get_answer(question["question"], ordered_alternatives)    if answer == correct_answer:
        print("⭐ Correct! ⭐")
        return 1
    else:
        print(f"The answer is {correct_answer!r}, not {answer!r}")
        return 0
```

现在，您可以从新的`question`字典中明确地挑选出问题文本、正确答案和备选答案。这样做的一个好处是，它比早期假设第一个答案是正确答案的惯例更具可读性。

您不需要在`get_answer()`中做任何修改，因为该函数已经处理了问题文本和一般的备选项列表。这一点没有改变。

您可以在下面折叠的部分中找到您的应用程序的当前完整源代码:



完整的`questions.toml`数据文件复制如下:

```py
# questions.toml [[questions]] question  =  "When was the first known use of the word 'quiz'" answer  =  "1781" alternatives  =  ["1771",  "1871",  "1881"] [[questions]] question  =  "Which built-in function can get information from the user" answer  =  "input" alternatives  =  ["get",  "print",  "write"] [[questions]] question  =  "What's the purpose of the built-in zip() function" answer  =  "To iterate over two or more sequences at the same time" alternatives  =  [ "To combine several strings into one", "To compress several files into one archive", "To get information from the user", ] [[questions]] question  =  "What does dict.get(key) return if key isn't found in dict" answer  =  "None" alternatives  =  ["key",  "True",  "False"] [[questions]] question  =  "How do you iterate over both indices and elements in an iterable" answer  =  "enumerate(iterable)" alternatives  =  [ "enumerate(iterable, start=1)", "range(iterable)", "range(iterable, start=1)", ] [[questions]] question  =  "What's the official name of the := operator" answer  =  "Assignment expression" alternatives  =  ["Named expression",  "Walrus operator",  "Colon equals operator"] [[questions]] question  =  "What's one effect of calling random.seed(42)" answer  =  "The random numbers are reproducible." alternatives  =  [ "The random numbers are more random.", "The computer clock is reset.", "The first random number is always 42.", ] [[questions]] question  =  "When does __name__ == '__main__' equal True in a Python file" answer  =  "When the file is run as a script" alternatives  =  [ "When the file is imported as a module", "When the file has a valid name", "When the file only has one function", ] [[questions]] question  =  "Which version of Python is the first with TOML support built in" answer  =  "3.11" alternatives  =  ["3.9",  "3.10",  "3.12"] [[questions]] question  =  "What's the name of the list-like data structure in TOML" answer  =  "Array" alternatives  =  ["List",  "Sequence",  "Set"]
```

将该文件保存在与`quiz.py`相同的文件夹中。



下面列出了测验应用程序的完整源代码:

```py
# quiz.py

import pathlib
import random
from string import ascii_lowercase
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

NUM_QUESTIONS_PER_QUIZ = 5
QUESTIONS_PATH = pathlib.Path(__file__).parent / "questions.toml"

def run_quiz():
    questions = prepare_questions(
        QUESTIONS_PATH, num_questions=NUM_QUESTIONS_PER_QUIZ
    )

    num_correct = 0
    for num, question in enumerate(questions, start=1):
        print(f"\nQuestion {num}:")
        num_correct += ask_question(question)

    print(f"\nYou got {num_correct} correct out of {num} questions")

def prepare_questions(path, num_questions):
    questions = tomllib.loads(path.read_text())["questions"]
    num_questions = min(num_questions, len(questions))
    return random.sample(questions, k=num_questions)

def ask_question(question):
    correct_answer = question["answer"]
    alternatives = [question["answer"]] + question["alternatives"]
    ordered_alternatives = random.sample(alternatives, k=len(alternatives))

    answer = get_answer(question["question"], ordered_alternatives)
    if answer == correct_answer:
        print("⭐ Correct! ⭐")
        return 1
    else:
        print(f"The answer is {correct_answer!r}, not {answer!r}")
        return 0

def get_answer(question, alternatives):
    print(f"{question}?")
    labeled_alternatives = dict(zip(ascii_lowercase, alternatives))
    for label, alternative in labeled_alternatives.items():
        print(f" {label}) {alternative}")

    while (answer_label := input("\nChoice? ")) not in labeled_alternatives:
        print(f"Please answer one of {', '.join(labeled_alternatives)}")

    return labeled_alternatives[answer_label]

if __name__ == "__main__":
    run_quiz()
```

使用`python quiz.py`运行您的应用程序。

您定义问题的新灵活格式为您提供了一些选项，可以为测验应用程序添加更多功能。在下一步中，您将深入了解其中的一些内容。

[*Remove ads*](/account/join/)

## 步骤 5:扩展您的测验功能

在第五步中，您将向测验应用程序添加更多功能。最后，您在前面的步骤中所做的重构将会得到回报！您将添加以下功能:

*   有多个正确答案的问题
*   可以指向正确答案的提示
*   可以作为教学时机的解释

在这一步结束时，您的应用程序将如下工作:

[https://player.vimeo.com/video/717554892?background=1](https://player.vimeo.com/video/717554892?background=1)

这些新功能为通过测验应用程序挑战自我的人提供了更有趣的体验。完成这一步后，您可以点击下方并进入`source_code_step_5`目录，查看应用程序的源代码:

**获取源代码:** [单击此处获取您将用于构建测验应用程序的源代码](https://realpython.com/bonus/python-quiz-application-project-code/)。

### 允许多个正确答案

有些问题可能有多个正确答案，如果你的测验也能回答这些问题，那就太好了。在本节中，您将添加对多个正确答案的支持。

首先，你需要考虑如何在你的`questions.toml`数据文件中表示几个正确的答案。您在上一步中介绍的更明确的数据结构的一个优点是，您也可以使用数组来指定正确的答案。将 TOML 文件中的每个`answer`键替换为一个`answers`键，将每个正确的答案放在方括号中(`[]`)。

您的问题文件将如下所示:

```py
# questions.toml [[questions]] question  =  "What's the name of the list-like data structure in TOML" answers  =  ["Array"]  alternatives  =  ["List",  "Sequence",  "Set"] [[questions]] question  =  "How can you run a Python script named quiz.py" answers  =  ["python quiz.py",  "python -m quiz"]  alternatives  =  ["python quiz",  "python -m quiz.py"]
```

对于只有一个正确答案的老问题，在`answers`数组中只会列出一个答案。上面的最后一个问题显示了一个有两个正确答案选项的问题示例。

一旦更新了数据结构，您还需要在代码中实现该特性。不需要对`run_quiz()`或者`prepare_questions()`做任何改动。在`ask_question()`中，你需要检查是否给出了所有的正确答案，而在`get_answer()`中，你需要能够阅读用户的多个答案。

从后一个挑战开始。用户如何输入多个答案，您如何验证每个答案都是有效的？一种可能是以逗号分隔的字符串形式输入多个答案。然后，您可以将字符串转换为列表，如下所示:

>>>

```py
>>> answer = "a,b, c"
>>> answer.replace(",", " ").split()
['a', 'b', 'c']
```

你可以使用`.split(",")`直接在逗号上分割。然而，首先用空格替换逗号，然后在缺省的空格上进行拆分，这增加了逗号周围允许空格的宽容度。这对你的用户来说会是一个更好的体验，因为他们可以不用逗号来写`a,b`、`a, b`，甚至`a b`，你的程序应该按照预期来解释它。

然而，有效答案的测试变得有点复杂。因此，你用一个更灵活的环替换了这个紧的环。为了循环直到得到一个有效的答案，您启动了一个无限循环，一旦所有的测试都通过，您就返回。将`get_answer()`重命名为`get_answers()`，并更新如下:

```py
# quiz.py

# ...

def get_answers(question, alternatives, num_choices=1):
    print(f"{question}?")
    labeled_alternatives = dict(zip(ascii_lowercase, alternatives))
    for label, alternative in labeled_alternatives.items():
        print(f" {label}) {alternative}")

    while True:
        plural_s = "" if num_choices == 1 else f"s (choose {num_choices})"
        answer = input(f"\nChoice{plural_s}? ")
        answers = set(answer.replace(",", " ").split())

        # Handle invalid answers
        if len(answers) != num_choices:
            plural_s = "" if num_choices == 1 else "s, separated by comma"
            print(f"Please answer {num_choices} alternative{plural_s}")
            continue

        if any(
            (invalid := answer) not in labeled_alternatives
            for answer in answers
        ):
            print(
                f"{invalid!r} is not a valid choice. "
                f"Please use {', '.join(labeled_alternatives)}"
            )
            continue

        return [labeled_alternatives[answer] for answer in answers]
```

在仔细查看代码中的细节之前，先测试一下这个函数:

>>>

```py
>>> from quiz import get_answers
>>> get_answers(
...     "Pick two numbers", ["one", "two", "three", "four"], num_choices=2
... )
Pick two numbers?
 a) one
 b) two
 c) three
 d) four

Choices (choose 2)? a Please answer 2 alternatives, separated by comma

Choices (choose 2)? d, e 'e' is not a valid choice. Please use a, b, c, d

Choices (choose 2)? d, b ['four', 'two']
```

您的函数首先检查答案是否包含适当数量的选项。然后检查每一个以确保它是一个有效的选择。如果这些检查中有任何一项失败，那么就会向用户打印一条有用的消息。

在代码中，当涉及到语法时，您还需要努力处理一个和几个项目之间的区别。您可以使用`plural_s`来修改文本字符串，以便在需要时包含多个 *s* 。

此外，您将答案转换为一个[集合](https://realpython.com/python-sets/)，以快速忽略重复的选项。类似于`"a, b, a"`的答案字符串被解释为`{"a", "b"}`。

最后，注意`get_answers()`返回一个字符串列表，而不是由`get_answer()`返回的普通字符串。

接下来，您使`ask_question()`适应多个正确答案的可能性。既然`get_answers()`已经处理了大部分的复杂问题，剩下的就是检查*所有的*答案，而不是只有一个。回想一下，`question`是一本包含关于一个问题的所有信息的字典，所以你不再需要通过`alternatives`。

因为答案的顺序无关紧要，所以在将给出的答案与正确答案进行比较时，可以使用`set()`:

```py
# quiz.py

# ...

def ask_question(question):
 correct_answers = question["answers"] alternatives = question["answers"] + question["alternatives"]    ordered_alternatives = random.sample(alternatives, k=len(alternatives))

 answers = get_answers(        question=question["question"],
        alternatives=ordered_alternatives,
 num_choices=len(correct_answers),    )
 if set(answers) == set(correct_answers):        print("⭐ Correct! ⭐")
        return 1
    else:
 is_or_are = " is" if len(correct_answers) == 1 else "s are" print("\n- ".join([f"No, the answer{is_or_are}:"] + correct_answers))        return 0
```

如果用户找到了所有的正确答案，你只能为他们赢得一分。否则，请列出所有正确答案。现在，您可以再次运行 Python 测验应用程序:

```py
$ python quiz.py

Question 1:
How can you run a Python script named quiz.py?
 a) python -m quiz
 b) python quiz
 c) python quiz.py
 d) python -m quiz.py

Choices (choose 2)? a Please answer 2 alternatives, separated by comma

Choices (choose 2)? a, c ⭐ Correct! ⭐

Question 2:
What's the name of the list-like data structure in TOML?
 a) Array
 b) Set
 c) Sequence
 d) List

Choice? e 'e' is not a valid choice. Please use a, b, c, d

Choice? c No, the answer is:
- Array

You got 1 correct out of 2 questions
```

允许多个正确答案可以让你在测验中更灵活地提问。

[*Remove ads*](/account/join/)

### 添加提示以帮助用户

有时候当你被问到一个问题时，你需要一点帮助来唤起你的记忆。给用户看到提示的选项可以让你的测验更有趣。在这一节中，您将扩展您的应用程序以包含**提示**。

您可以在您的`questions.toml`数据文件中包含提示，例如通过添加`hint`作为可选的键值对:

```py
# questions.toml [[questions]] question  =  "How can you run a Python script named quiz.py" answers  =  ["python quiz.py",  "python -m quiz"] alternatives  =  ["python quiz",  "python -m quiz.py"] hint  =  "One option uses the filename, and the other uses the module name."  
[[questions]] question  =  "What's a PEP" answers  =  ["A Python Enhancement Proposal"] alternatives  =  [ "A Pretty Exciting Policy", "A Preciously Evolved Python", "A Potentially Epic Prize", ] hint  =  "PEPs are used to evolve Python."
```

TOML 文件中的每个问题都由 Python 中的一个字典表示。新的`hint`字段在那些字典中显示为新的键。这样做的一个效果是，您不需要改变读取问题数据的方式，即使您对数据结构做了很小的更改。

相反，您可以修改代码以利用新的可选字段。在`ask_question()`中，你只需要做一个小小的改变:

```py
# quiz.py

# ...

def ask_question(question):
    # ...
    answers = get_answers(
        question=question["question"],
        alternatives=ordered_alternatives,
        num_choices=len(correct_answers),
 hint=question.get("hint"),    )
    # ...
```

你用`question.get("hint")`而不是`question["hint"]`，因为不是所有的问题都有提示。如果其中一个`question`字典没有将`"hint"`定义为一个键，那么`question.get("hint")`返回`None`，然后将其传递给`get_answers()`。

同样，您将对`get_answers()`进行更大的更改。您将使用特殊的问号(`?`)标签将提示添加为备选答案之一:

```py
# quiz.py

# ...

def get_answers(question, alternatives, num_choices=1, hint=None):
    print(f"{question}?")
    labeled_alternatives = dict(zip(ascii_lowercase, alternatives))
 if hint: labeled_alternatives["?"] = "Hint" 
    for label, alternative in labeled_alternatives.items():
        print(f" {label}) {alternative}")

    while True:
        plural_s = "" if num_choices == 1 else f"s (choose {num_choices})"
        answer = input(f"\nChoice{plural_s}? ")
        answers = set(answer.replace(",", " ").split())

        # Handle hints
 if hint and "?" in answers: print(f"\nHINT: {hint}") continue 
        # Handle invalid answers
        # ...

        return [labeled_alternatives[answer] for answer in answers]
```

如果提供了提示，则将其添加到`labeled_alternatives`的末尾。然后，用户可以使用`?`查看打印到屏幕上的提示。如果您测试您的测验应用程序，那么您现在会得到一些友好的帮助:

```py
$ python quiz.py

Question 1:
What's a PEP?
 a) A Potentially Epic Prize
 b) A Preciously Evolved Python
 c) A Python Enhancement Proposal
 d) A Pretty Exciting Policy
 ?) Hint

Choice? ? 
HINT: PEPs are used to evolve Python.

Choice? c ⭐ Correct! ⭐
```

在下一节中，您将添加一个类似的特性。除了在用户回答问题之前显示可选提示之外，您还将在用户回答问题之后显示解释。

### 添加解释以强化学习

你可以实现**解释**，就像你在上一节中实现提示一样。首先，您将在数据文件中添加一个可选的`explanation`字段。然后，在您的应用程序中，您将在用户回答问题后显示解释。

从在`questions.toml`中添加`explanation`键开始:

```py
# questions.toml [[questions]] question  =  "What's a PEP" answers  =  ["A Python Enhancement Proposal"] alternatives  =  [ "A Pretty Exciting Policy", "A Preciously Evolved Python", "A Potentially Epic Prize", ] hint  =  "PEPs are used to evolve Python." explanation  =  """
 Python Enhancement Proposals (PEPs) are design documents that provide information to the Python community. PEPs are used to propose new features for the Python language, to collect community input on an issue, and to document design decisions made about the language. """  
[[questions]] question  =  "How can you add a docstring to a function" answers  =  [ "By writing a string literal as the first statement in the function", "By assigning a string to the function's .__doc__ attribute", ] alternatives  =  [ "By using the built-in @docstring decorator", "By returning a string from the function", ] hint  =  "They're parsed from your code and stored on the function object." explanation  =  """
 Docstrings document functions and other Python objects. A docstring is a string literal that occurs as the first statement in a module, function, class, or method definition. Such a docstring becomes the .__doc__ special attribute of that object. See PEP 257 for more information.   There is no built-in @docstring decorator. Many functions naturally return strings. Such a feature can therefore not be used for docstrings. """
```

TOML 通过像 Python 一样使用三重引号(`"""`)来支持[多行字符串](https://realpython.com/python-comments-guide/#python-multiline-comments)。这对于可能跨越几个句子的解释非常有用。

用户回答问题后，解释将打印到屏幕上。换句话说，解释不是在`get_answers()`中完成的用户交互的一部分。相反，您将在`ask_question()`中打印它们:

```py
# quiz.py

# ...

def ask_question(question):
    correct_answers = question["answers"]
    alternatives = question["answers"] + question["alternatives"]
    ordered_alternatives = random.sample(alternatives, k=len(alternatives))

    answers = get_answers(
        question=question["question"],
        alternatives=ordered_alternatives,
        num_choices=len(correct_answers),
        hint=question.get("hint"),
    )
    if correct := (set(answers) == set(correct_answers)):
        print("⭐ Correct! ⭐")
    else:
        is_or_are = " is" if len(correct_answers) == 1 else "s are"
        print("\n- ".join([f"No, the answer{is_or_are}:"] + correct_answers))

 if "explanation" in question: print(f"\nEXPLANATION:\n{question['explanation']}") 
 return 1 if correct else 0
```

因为您在向用户反馈他们的答案是否正确后打印了解释，所以您不能再返回到`if` … `else`块内。你因此把 [`return`](https://realpython.com/python-return-statement/) 语句移到了函数的末尾。

当您运行测验应用程序时，您的解释如下所示:

```py
$ python quiz.py

Question 1:
How can you add a docstring to a function?
 a) By returning a string from the function
 b) By assigning a string to the function's .__doc__ attribute
 c) By writing a string literal as the first statement in the function
 d) By using the built-in @docstring decorator
 ?) Hint

Choices (choose 2)? a, b No, the answers are:
- By writing a string literal as the first statement in the function
- By assigning a string to the function's .__doc__ attribute

EXPLANATION:
 Docstrings document functions and other Python objects. A docstring is a
 string literal that occurs as the first statement in a module, function,
 class, or method definition. Such a docstring becomes the .__doc__ special
 attribute of that object. See PEP 257 for more information.

 There is no built-in @docstring decorator. Many functions naturally return
 strings. Such a feature can therefore not be used for docstrings.
```

Python 测验应用程序的改进是累积的。请随意展开下面折叠的部分，查看包含所有新特性的完整源代码:



完整的`questions.toml`数据文件复制如下:

```py
# questions.toml [[questions]] question  =  "When was the first known use of the word 'quiz'" answers  =  ["1781"] alternatives  =  ["1771",  "1871",  "1881"] [[questions]] question  =  "Which built-in function can get information from the user" answers  =  ["input"] alternatives  =  ["get",  "print",  "write"] [[questions]] question  =  "What's the purpose of the built-in zip() function" answers  =  ["To iterate over two or more sequences at the same time"] alternatives  =  [ "To combine several strings into one", "To compress several files into one archive", "To get information from the user", ] [[questions]] question  =  "What does dict.get(key) return if key isn't found in dict" answers  =  ["None"] alternatives  =  ["key",  "True",  "False"] [[questions]] question  =  "How do you iterate over both indices and elements in an iterable" answers  =  ["enumerate(iterable)"] alternatives  =  [ "enumerate(iterable, start=1)", "range(iterable)", "range(iterable, start=1)", ] [[questions]] question  =  "What's the official name of the := operator" answers  =  ["Assignment expression"] alternatives  =  ["Named expression",  "Walrus operator",  "Colon equals operator"] [[questions]] question  =  "What's one effect of calling random.seed(42)" answers  =  ["The random numbers are reproducible."] alternatives  =  [ "The random numbers are more random.", "The computer clock is reset.", "The first random number is always 42.", ] [[questions]] question  =  "When does __name__ == '__main__' equal True in a Python file" answers  =  ["When the file is run as a script"] alternatives  =  [ "When the file is imported as a module", "When the file has a valid name", "When the file only has one function", ] [[questions]] question  =  "Which version of Python is the first with TOML support built in" answers  =  ["3.11"] alternatives  =  ["3.9",  "3.10",  "3.12"] [[questions]] question  =  "What's the name of the list-like data structure in TOML" answers  =  ["Array"] alternatives  =  ["List",  "Sequence",  "Set"] [[questions]] question  =  "How can you run a Python script named quiz.py" answers  =  ["python quiz.py",  "python -m quiz"] alternatives  =  ["python quiz",  "python -m quiz.py"] hint  =  "One option uses the filename, and the other uses the module name." [[questions]] question  =  "What's a PEP" answers  =  ["A Python Enhancement Proposal"] alternatives  =  [ "A Pretty Exciting Policy", "A Preciously Evolved Python", "A Potentially Epic Prize", ] hint  =  "PEPs are used to evolve Python." explanation  =  """
 Python Enhancement Proposals (PEPs) are design documents that provide
 information to the Python community. PEPs are used to propose new features
 for the Python language, to collect community input on an issue, and to
 document design decisions made about the language.
""" [[questions]] question  =  "How can you add a docstring to a function" answers  =  [ "By writing a string literal as the first statement in the function", "By assigning a string to the function's .__doc__ attribute", ] alternatives  =  [ "By using the built-in @docstring decorator", "By returning a string from the function", ] hint  =  "They are parsed from your code and stored on the function object." explanation  =  """
 Docstrings document functions and other Python objects. A docstring is a
 string literal that occurs as the first statement in a module, function,
 class, or method definition. Such a docstring becomes the .__doc__ special
 attribute of that object. See PEP 257 for more information.

 There is no built-in @docstring decorator. Many functions naturally return
 strings. Such a feature can therefore not be used for docstrings.
"""
```

将该文件保存在与`quiz.py`相同的文件夹中。



下面列出了测验应用程序的完整源代码:

```py
# quiz.py

import pathlib
import random
from string import ascii_lowercase
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

NUM_QUESTIONS_PER_QUIZ = 5
QUESTIONS_PATH = pathlib.Path(__file__).parent / "questions.toml"

def run_quiz():
    questions = prepare_questions(
        QUESTIONS_PATH, num_questions=NUM_QUESTIONS_PER_QUIZ
    )

    num_correct = 0
    for num, question in enumerate(questions, start=1):
        print(f"\nQuestion {num}:")
        num_correct += ask_question(question)

    print(f"\nYou got {num_correct} correct out of {num} questions")

def prepare_questions(path, num_questions):
    questions = tomllib.loads(path.read_text())["questions"]
    num_questions = min(num_questions, len(questions))
    return random.sample(questions, k=num_questions)

def ask_question(question):
    correct_answers = question["answers"]
    alternatives = question["answers"] + question["alternatives"]
    ordered_alternatives = random.sample(alternatives, k=len(alternatives))

    answers = get_answers(
        question=question["question"],
        alternatives=ordered_alternatives,
        num_choices=len(correct_answers),
        hint=question.get("hint"),
    )
    if correct := (set(answers) == set(correct_answers)):
        print("⭐ Correct! ⭐")
    else:
        is_or_are = " is" if len(correct_answers) == 1 else "s are"
        print("\n- ".join([f"No, the answer{is_or_are}:"] + correct_answers))

    if "explanation" in question:
        print(f"\nEXPLANATION:\n{question['explanation']}")

    return 1 if correct else 0

def get_answers(question, alternatives, num_choices=1, hint=None):
    print(f"{question}?")
    labeled_alternatives = dict(zip(ascii_lowercase, alternatives))
    if hint:
        labeled_alternatives["?"] = "Hint"

    for label, alternative in labeled_alternatives.items():
        print(f" {label}) {alternative}")

    while True:
        plural_s = "" if num_choices == 1 else f"s (choose {num_choices})"
        answer = input(f"\nChoice{plural_s}? ")
        answers = set(answer.replace(",", " ").split())

        # Handle hints
        if hint and "?" in answers:
            print(f"\nHINT: {hint}")
            continue

        # Handle invalid answers
        if len(answers) != num_choices:
            plural_s = "" if num_choices == 1 else "s, separated by comma"
            print(f"Please answer {num_choices} alternative{plural_s}")
            continue

        if any(
            (invalid := answer) not in labeled_alternatives
            for answer in answers
        ):
            print(
                f"{invalid!r} is not a valid choice. "
                f"Please use {', '.join(labeled_alternatives)}"
            )
            continue

        return [labeled_alternatives[answer] for answer in answers]

if __name__ == "__main__":
    run_quiz()
```

使用`python quiz.py`运行您的应用程序。

在最后一步，您将添加另一个特性:在您的应用程序中支持几个测验主题。

## 第六步:支持几个测验题目

在本节中，您将进行最后一项改进，这将使您的 Python 测验应用程序更加有趣、多样和有趣。您将添加将问题分组到不同主题的选项，并让您的用户选择他们将被提问的主题。

Python 测验应用程序的最终版本将如下所示:

[https://player.vimeo.com/video/717554866?background=1](https://player.vimeo.com/video/717554866?background=1)

更多的主题和新问题将使您的测验申请保持新鲜。点击下面并导航到`source_code_final`目录，查看添加这些内容后源代码的外观:

**获取源代码:** [单击此处获取您将用于构建测验应用程序的源代码](https://realpython.com/bonus/python-quiz-application-project-code/)。

TOML 文件中的节可以嵌套。您可以通过在节标题中添加句点(`.`)来创建嵌套表。作为一个说明性的例子，考虑下面的 TOML 文档:

>>>

```py
>>> toml = """
... [python]
... label = "Python"
... ... [python.version]
... number = "3.10"
... release.date = 2021-10-04
... release.manager = "@pyblogsal"
... """

>>> import tomli
>>> tomli.loads(toml)
{'python': {'label': 'Python', 'version': {
 'release': {'date': datetime.date(2021, 10, 4), 'manager': '@pyblogsal'},
 'number': '3.10'}}}
```

这里，节头`[python.version]`被表示为嵌套在`python`内的`version`。类似地，带句点的键也被解释为嵌套字典，如本例中的`release`所示。

您可以重新组织`questions.toml`,为每个主题包含一个部分。除了嵌套的`questions`数组，您将添加一个`label`键，为每个主题提供一个名称。更新您的数据文件以使用以下格式:

```py
# questions.toml [python] label  =  "Python" [[python.questions]] question  =  "How can you add a docstring to a function" answers  =  [ "By writing a string literal as the first statement in the function", "By assigning a string to the function's .__doc__ attribute", ] alternatives  =  [ "By using the built-in @docstring decorator", "By returning a string from the function", ] hint  =  "They're parsed from your code and stored on the function object." explanation  =  """
 Docstrings document functions and other Python objects. A docstring is a
 string literal that occurs as the first statement in a module, function,
 class, or method definition. Such a docstring becomes the .__doc__ special
 attribute of that object. See PEP 257 for more information.

 There's no built-in @docstring decorator. Many functions naturally return
 strings. Such a feature can therefore not be used for docstrings.
""" [[python.questions]] question  =  "When was the first public version of Python released?" answers  =  ["February 1991"] alternatives  =  ["January 1994",  "October 2000",  "December 2008"] hint  =  "The first public version was labeled version 0.9.0." explanation  =  """
 Guido van Rossum started work on Python in December 1989\. He posted
 Python v0.9.0 to the alt.sources newsgroup in February 1991\. Python
 reached version 1.0.0 in January 1994\. The next major versions,
 Python 2.0 and Python 3.0, were released in October 2000 and December
 2008, respectively.
""" [capitals] label  =  "Capitals" [[capitals.questions]] question  =  "What's the capital of Norway" answers  =  ["Oslo"] hint  =  "Lars Onsager, Jens Stoltenberg, Trygve Lie, and Børge Ousland." alternatives  =  ["Stockholm",  "Copenhagen",  "Helsinki",  "Reykjavik"] explanation  =  """
 Oslo was founded as a city in the 11th century and established as a
 trading place. It became the capital of Norway in 1299\. The city was
 destroyed by a fire in 1624 and rebuilt as Christiania, named in honor
 of the reigning king. The city was renamed back to Oslo in 1925.
""" [[capitals.questions]] question  =  "What's the state capital of Texas, USA" answers  =  ["Austin"] alternatives  =  ["Harrisburg",  "Houston",  "Galveston",  "Columbia"] hint  =  "SciPy is held there each year." explanation  =  """
 Austin is named in honor of Stephen F. Austin. It was purpose-built to
 be the capital of Texas and was incorporated in December 1839\. Houston,
 Harrisburg, Columbia, and Galveston are all earlier capitals of Texas.
"""
```

现在，数据文件中包含了两个主题:Python 和 Capitals。在每个主题部分中，问题表的结构仍然和以前一样。这意味着你需要做的唯一改变就是你准备问题的方式。

你从阅读和解析`questions.toml`开始。接下来，您挑选出每个主题并将其存储在一个新的临时字典中。你需要问用户他们想尝试哪个话题。幸运的是，您可以重用`get_answers()`来获得这方面的输入。最后，你挑出属于所选主题的问题，并把它们混在一起:

```py
# quiz.py

# ...

def prepare_questions(path, num_questions):
 topic_info = tomllib.loads(path.read_text()) topics = { topic["label"]: topic["questions"] for topic in topic_info.values() } topic_label = get_answers( question="Which topic do you want to be quizzed about", alternatives=sorted(topics), )[0]   questions = topics[topic_label]    num_questions = min(num_questions, len(questions))
    return random.sample(questions, k=num_questions)
```

`prepare_questions()`返回的数据结构仍然和以前一样，所以不需要对`run_quiz()`、`ask_question()`或`get_answers()`做任何修改。当这些类型的更新只需要您编辑一个或几个函数时，这是一个好的迹象，表明您的代码结构良好，具有良好的抽象。

运行 Python 测试应用程序。你会看到新的主题提示:

```py
$ python quiz.py
Which topic do you want to be quizzed about?
 a) Capitals
 b) Python

Choice? a 
Question 1:
What's the capital of Norway?
 a) Reykjavik
 b) Helsinki
 c) Stockholm
 d) Copenhagen
 e) Oslo
 ?) Hint

Choice? ? 
HINT: Lars Onsager, Jens Stoltenberg, Trygve Lie, and Børge Ousland.

Choice? e ⭐ Correct! ⭐

EXPLANATION:
 Oslo was founded as a city in the 11th century and established as a
 trading place. It became the capital of Norway in 1299\. The city was
 destroyed by a fire in 1624 and rebuilt as Christiania, named in honor
 of the reigning king. The city was renamed back to Oslo in 1925.
```

这就结束了这个旅程的引导部分。您已经在终端中创建了一个强大的 Python 测验应用程序。您可以通过展开下面的框来查看完整的源代码以及问题列表:



完整的`questions.toml`数据文件复制如下:

```py
# questions.toml [python] label  =  "Python" [[python.questions]] question  =  "When was the first known use of the word 'quiz'" answers  =  ["1781"] alternatives  =  ["1771",  "1871",  "1881"] [[python.questions]] question  =  "Which built-in function can get information from the user" answers  =  ["input"] alternatives  =  ["get",  "print",  "write"] [[python.questions]] question  =  "What's the purpose of the built-in zip() function" answers  =  ["To iterate over two or more sequences at the same time"] alternatives  =  [ "To combine several strings into one", "To compress several files into one archive", "To get information from the user", ] [[python.questions]] question  =  "What does dict.get(key) return if key isn't found in dict" answers  =  ["None"] alternatives  =  ["key",  "True",  "False"] [[python.questions]] question  =  "How do you iterate over both indices and elements in an iterable" answers  =  ["enumerate(iterable)"] alternatives  =  [ "enumerate(iterable, start=1)", "range(iterable)", "range(iterable, start=1)", ] [[python.questions]] question  =  "What's the official name of the := operator" answers  =  ["Assignment expression"] alternatives  =  [ "Named expression", "Walrus operator", "Colon equals operator", ] [[python.questions]] question  =  "What's one effect of calling random.seed(42)" answers  =  ["The random numbers are reproducible."] alternatives  =  [ "The random numbers are more random.", "The computer clock is reset.", "The first random number is always 42.", ] [[python.questions]] question  =  "Which version of Python is the first with TOML support built in" answers  =  ["3.11"] alternatives  =  ["3.9",  "3.10",  "3.12"] [[python.questions]] question  =  "How can you run a Python script named quiz.py" answers  =  ["python quiz.py",  "python -m quiz"] alternatives  =  ["python quiz",  "python -m quiz.py"] hint  =  "One option uses the filename, and the other uses the module name." [[python.questions]] question  =  "What's the name of the list-like data structure in TOML" answers  =  ["Array"] alternatives  =  ["List",  "Sequence",  "Set"] [[python.questions]] question  =  "What's a PEP" answers  =  ["A Python Enhancement Proposal"] alternatives  =  [ "A Pretty Exciting Policy", "A Preciously Evolved Python", "A Potentially Epic Prize", ] hint  =  "PEPs are used to evolve Python." explanation  =  """
Python Enhancement Proposals (PEPs) are design documents that provide
information to the Python community. PEPs are used to propose new features
for the Python language, to collect community input on an issue, and to
document design decisions made about the language.
""" [[python.questions]] question  =  "How can you add a docstring to a function" answers  =  [ "By writing a string literal as the first statement in the function", "By assigning a string to the function's .__doc__ attribute", ] alternatives  =  [ "By using the built-in @docstring decorator", "By returning a string from the function", ] hint  =  "They are parsed from your code and stored on the function object." explanation  =  """
Docstrings document functions and other Python objects. A docstring is a
string literal that occurs as the first statement in a module, function,
class, or method definition. Such a docstring becomes the .__doc__ special
attribute of that object. See PEP 257 for more information.

There's no built-in @docstring decorator. Many functions naturally return
strings. Such a feature can therefore not be used for docstrings.
""" [[python.questions]] question  =  "When was the first public version of Python released" answers  =  ["February 1991"] alternatives  =  ["January 1994",  "October 2000",  "December 2008"] hint  =  "The first public version was labeled version 0.9.0." explanation  =  """
Guido van Rossum started work on Python in December 1989\. He posted
Python v0.9.0 to the alt.sources newsgroup in February 1991\. Python
reached version 1.0.0 in January 1994\. The next major versions,
Python 2.0 and Python 3.0, were released in October 2000 and December
2008, respectively.
""" [capitals] label  =  "Capitals" [[capitals.questions]] question  =  "What's the capital of Norway" answers  =  ["Oslo"] hint  =  "Lars Onsager, Jens Stoltenberg, Trygve Lie, and Børge Ousland." alternatives  =  ["Stockholm",  "Copenhagen",  "Helsinki",  "Reykjavik"] explanation  =  """
Oslo was founded as a city in the 11th century and established as a
trading place. It became the capital of Norway in 1299\. The city was
destroyed by a fire in 1624 and rebuilt as Christiania, named in honor
of the reigning king. The city was renamed back to Oslo in 1925.
""" [[capitals.questions]] question  =  "What's the state capital of Texas, USA" answers  =  ["Austin"] alternatives  =  ["Harrisburg",  "Houston",  "Galveston",  "Columbia"] hint  =  "SciPy is held there each year." explanation  =  """
Austin is named in honor of Stephen F. Austin. It was purpose-built to
be the capital of Texas and was incorporated in December 1839\. Houston,
Harrisburg, Columbia, and Galveston are all earlier capitals of Texas.
"""
```

将该文件保存在与`quiz.py`相同的文件夹中。



下面列出了您的测验应用程序的完整源代码:

```py
# quiz.py

import pathlib
import random
from string import ascii_lowercase
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

NUM_QUESTIONS_PER_QUIZ = 5
QUESTIONS_PATH = pathlib.Path(__file__).parent / "questions.toml"

def run_quiz():
    questions = prepare_questions(
        QUESTIONS_PATH, num_questions=NUM_QUESTIONS_PER_QUIZ
    )

    num_correct = 0
    for num, question in enumerate(questions, start=1):
        print(f"\nQuestion {num}:")
        num_correct += ask_question(question)

    print(f"\nYou got {num_correct} correct out of {num} questions")

def prepare_questions(path, num_questions):
    topic_info = tomllib.loads(path.read_text())
    topics = {
        topic["label"]: topic["questions"] for topic in topic_info.values()
    }
    topic_label = get_answers(
        question="Which topic do you want to be quizzed about",
        alternatives=sorted(topics),
    )[0]

    questions = topics[topic_label]
    num_questions = min(num_questions, len(questions))
    return random.sample(questions, k=num_questions)

def ask_question(question):
    correct_answers = question["answers"]
    alternatives = question["answers"] + question["alternatives"]
    ordered_alternatives = random.sample(alternatives, k=len(alternatives))

    answers = get_answers(
        question=question["question"],
        alternatives=ordered_alternatives,
        num_choices=len(correct_answers),
        hint=question.get("hint"),
    )
    if correct := (set(answers) == set(correct_answers)):
        print("⭐ Correct! ⭐")
    else:
        is_or_are = " is" if len(correct_answers) == 1 else "s are"
        print("\n- ".join([f"No, the answer{is_or_are}:"] + correct_answers))

    if "explanation" in question:
        print(f"\nEXPLANATION:\n{question['explanation']}")

    return 1 if correct else 0

def get_answers(question, alternatives, num_choices=1, hint=None):
    print(f"{question}?")
    labeled_alternatives = dict(zip(ascii_lowercase, alternatives))
    if hint:
        labeled_alternatives["?"] = "Hint"

    for label, alternative in labeled_alternatives.items():
        print(f" {label}) {alternative}")

    while True:
        plural_s = "" if num_choices == 1 else f"s (choose {num_choices})"
        answer = input(f"\nChoice{plural_s}? ")
        answers = set(answer.replace(",", " ").split())

        # Handle hints
        if hint and "?" in answers:
            print(f"\nHINT: {hint}")
            continue

        # Handle invalid answers
        if len(answers) != num_choices:
            plural_s = "" if num_choices == 1 else "s, separated by comma"
            print(f"Please answer {num_choices} alternative{plural_s}")
            continue

        if any(
            (invalid := answer) not in labeled_alternatives
            for answer in answers
        ):
            print(
                f"{invalid!r} is not a valid choice. "
                f"Please use {', '.join(labeled_alternatives)}"
            )
            continue

        return [labeled_alternatives[answer] for answer in answers]

if __name__ == "__main__":
    run_quiz()
```

使用`python quiz.py`运行您的应用程序。

您也可以通过单击下面的链接访问源代码和问题文件:

**获取源代码:** [单击此处获取您将用于构建测验应用程序的源代码](https://realpython.com/bonus/python-quiz-application-project-code/)。

您将在目录`source_code_final`中找到应用程序的最终版本。

## 结论

干得好！您已经用 Python 创建了一个灵活而有用的测验应用程序。在这个过程中，您已经了解了如何从一个基本脚本开始，然后将它构建成一个更复杂的程序。

**在本教程中，您已经学会了如何:**

*   **在终端与用户交互**
*   **提高**应用程序的可用性
*   **重构**你的应用程序，不断改进它
*   **将**数据存储在专用数据文件中

现在，去玩你的测验应用程序吧。自己补充一些问题，向朋友挑战。在下面的评论中分享你最好的问题和测验主题！

## 接下来的步骤

在本教程中，您已经创建了一个功能完善的测验应用程序。然而，这个项目仍然有很多改进的机会。

以下是一些关于附加功能的想法:

*   **测验创建者:**添加一个独立的应用程序，它可以交互地询问问题和答案，并以适当的 TOML 格式存储它们。
*   **在数据库中存储数据:**用[合适的数据库](https://realpython.com/python-mysql/)替换 TOML 数据文件。
*   **问题中心:**在线创建一个你的应用程序可以连接的中央问题数据库。
*   **多用户挑战:**允许不同用户在一场琐事比赛中互相挑战。

您还可以重用这个测验应用程序中的逻辑，但是要改变前端表示层。也许你可以将这个项目转换成一个网络应用程序或者创建一个 T2 的抽认卡应用程序来帮助你准备考试。欢迎在下面的评论中分享你的进步。****************