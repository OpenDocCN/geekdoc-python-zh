# 用 Python 漂亮的字体美化你的数据结构

> 原文：<https://realpython.com/python-pretty-print/>

处理数据对任何 Pythonista 来说都是必不可少的，但有时这些数据并不漂亮。计算机不关心格式，但是没有好的格式，人类可能会发现一些难以阅读的东西。当你在大字典或长列表上使用`print()`时，输出并不漂亮——它是有效的，但并不漂亮。

Python 中的`pprint`模块是一个实用模块，可以用来以一种可读的、**漂亮的**方式打印数据结构。它是标准库的一部分，对于调试处理 API 请求、大型 JSON 文件和一般数据的代码特别有用。

**本教程结束时，您将:**

*   了解**为什么**的`pprint`模块是**必需的**
*   学习如何使用 **`pprint()`** 、 **`PrettyPrinter`** ，以及它们的**参数**
*   能够创建自己的 **`PrettyPrinter`** 实例
*   保存**格式的字符串输出**而不是打印它
*   打印和识别**递归数据结构**

在这个过程中，您还会看到对公共 API 的 HTTP 请求和正在运行的 [JSON 解析](https://realpython.com/python-json/)。

**免费奖励:** 并学习 Python 3 的基础知识，如使用数据类型、字典、列表和 Python 函数。

## 了解 Python 漂亮打印的需求

Python `pprint`模块在很多情况下都很有用。当发出 API 请求、处理 [JSON 文件](https://realpython.com/python-json/)或处理复杂的嵌套数据时，它会派上用场。您可能会发现使用普通的 [`print()`](https://realpython.com/python-print/) 函数不足以有效地探索您的数据和[调试](https://realpython.com/python-debugging-pdb/)您的应用程序。当您将`print()`与[字典](https://realpython.com/python-dicts/)和[列表](https://realpython.com/python-lists-tuples/)一起使用时，输出不包含任何换行符。

在您开始探索`pprint`之前，您将首先使用`urllib`请求获取一些数据。您将向 [{JSON}占位符](https://jsonplaceholder.typicode.com/)请求一些模拟用户信息。首先要做的是发出 HTTP `GET`请求，并将响应放入字典中:

>>>

```py
>>> from urllib import request
>>> response = request.urlopen("https://jsonplaceholder.typicode.com/users")
>>> json_response = response.read()
>>> import json
>>> users = json.loads(json_response)
```

这里，您发出一个基本的`GET`请求，然后用`json.loads()`将响应解析到一个字典中。现在字典在一个变量中，下一步通常是用`print()`打印内容:

>>>

```py
>>> print(users)
[{'id': 1, 'name': 'Leanne Graham', 'username': 'Bret', 'email': 'Sincere@april.biz', 'address': {'street': 'Kulas Light', 'suite': 'Apt. 556', 'city': 'Gwenborough', 'zipcode': '92998-3874', 'geo': {'lat': '-37.3159', 'lng': '81.1496'}}, 'phone': '1-770-736-8031 x56442', 'website': 'hildegard.org', 'company': {'name': 'Romaguera-Crona', 'catchPhrase': 'Multi-layered client-server neural-net', 'bs': 'harness real-time e-markets'}}, {'id': 2, 'name': 'Ervin Howell', 'username': 'Antonette', 'email': 'Shanna@melissa.tv', 'address': {'street': 'Victor Plains', 'suite': 'Suite 879', 'city': 'Wisokyburgh', 'zipcode': '90566-7771', 'geo': {'lat': '-43.9509', 'lng': '-34.4618'}}, 'phone': '010-692-6593 x09125', 'website': 'anastasia.net', 'company': {'name': 'Deckow-Crist', 'catchPhrase': 'Proactive didactic contingency', 'bs': 'synergize scalable supply-chains'}}, {'id': 3, 'name': 'Clementine Bauch', 'username': 'Samantha', 'email': 'Nathan@yesenia.net', 'address': {'street': 'Douglas Extension', 'suite': 'Suite 847', 'city': 'McKenziehaven', 'zipcode': '59590-4157', 'geo': {'lat': '-68.6102', 'lng': '-47.0653'}}, 'phone': '1-463-123-4447', 'website': 'ramiro.info', 'company': {'name': 'Romaguera-Jacobson', 'catchPhrase': 'Face to face bifurcated interface', 'bs': 'e-enable strategic applications'}}, {'id': 4, 'name': 'Patricia Lebsack', 'username': 'Karianne', 'email': 'Julianne.OConner@kory.org', 'address': {'street': 'Hoeger Mall', 'suite': 'Apt. 692', 'city': 'South Elvis', 'zipcode': '53919-4257', 'geo': {'lat': '29.4572', 'lng': '-164.2990'}}, 'phone': '493-170-9623 x156', 'website': 'kale.biz', 'company': {'name': 'Robel-Corkery', 'catchPhrase': 'Multi-tiered zero tolerance productivity', 'bs': 'transition cutting-edge web services'}}, {'id': 5, 'name': 'Chelsey Dietrich', 'username': 'Kamren', 'email': 'Lucio_Hettinger@annie.ca', 'address': {'street': 'Skiles Walks', 'suite': 'Suite 351', 'city': 'Roscoeview', 'zipcode': '33263', 'geo': {'lat': '-31.8129', 'lng': '62.5342'}}, 'phone': '(254)954-1289', 'website': 'demarco.info', 'company': {'name': 'Keebler LLC', 'catchPhrase': 'User-centric fault-tolerant solution', 'bs': 'revolutionize end-to-end systems'}}, {'id': 6, 'name': 'Mrs. Dennis Schulist', 'username': 'Leopoldo_Corkery', 'email': 'Karley_Dach@jasper.info', 'address': {'street': 'Norberto Crossing', 'suite': 'Apt. 950', 'city': 'South Christy', 'zipcode': '23505-1337', 'geo': {'lat': '-71.4197', 'lng': '71.7478'}}, 'phone': '1-477-935-8478 x6430', 'website': 'ola.org', 'company': {'name': 'Considine-Lockman', 'catchPhrase': 'Synchronised bottom-line interface', 'bs': 'e-enable innovative applications'}}, {'id': 7, 'name': 'Kurtis Weissnat', 'username': 'Elwyn.Skiles', 'email': 'Telly.Hoeger@billy.biz', 'address': {'street': 'Rex Trail', 'suite': 'Suite 280', 'city': 'Howemouth', 'zipcode': '58804-1099', 'geo': {'lat': '24.8918', 'lng': '21.8984'}}, 'phone': '210.067.6132', 'website': 'elvis.io', 'company': {'name': 'Johns Group', 'catchPhrase': 'Configurable multimedia task-force', 'bs': 'generate enterprise e-tailers'}}, {'id': 8, 'name': 'Nicholas Runolfsdottir V', 'username': 'Maxime_Nienow', 'email': 'Sherwood@rosamond.me', 'address': {'street': 'Ellsworth Summit', 'suite': 'Suite 729', 'city': 'Aliyaview', 'zipcode': '45169', 'geo': {'lat': '-14.3990', 'lng': '-120.7677'}}, 'phone': '586.493.6943 x140', 'website': 'jacynthe.com', 'company': {'name': 'Abernathy Group', 'catchPhrase': 'Implemented secondary concept', 'bs': 'e-enable extensible e-tailers'}}, {'id': 9, 'name': 'Glenna Reichert', 'username': 'Delphine', 'email': 'Chaim_McDermott@dana.io', 'address': {'street': 'Dayna Park', 'suite': 'Suite 449', 'city': 'Bartholomebury', 'zipcode': '76495-3109', 'geo': {'lat': '24.6463', 'lng': '-168.8889'}}, 'phone': '(775)976-6794 x41206', 'website': 'conrad.com', 'company': {'name': 'Yost and Sons', 'catchPhrase': 'Switchable contextually-based project', 'bs': 'aggregate real-time technologies'}}, {'id': 10, 'name': 'Clementina DuBuque', 'username': 'Moriah.Stanton', 'email': 'Rey.Padberg@karina.biz', 'address': {'street': 'Kattie Turnpike', 'suite': 'Suite 198', 'city': 'Lebsackbury', 'zipcode': '31428-2261', 'geo': {'lat': '-38.2386', 'lng': '57.2232'}}, 'phone': '024-648-3804', 'website': 'ambrose.net', 'company': {'name': 'Hoeger LLC', 'catchPhrase': 'Centralized empowering task-force', 'bs': 'target end-to-end models'}}]
```

哦亲爱的！一大行没有换行符。根据您的控制台设置，这可能会显示为很长的一行。或者，您的控制台输出可能打开了自动换行模式，这是最常见的情况。不幸的是，这并没有使输出更加友好！

如果您查看第一个和最后一个字符，您可以看到这似乎是一个列表。您可能想开始编写一个循环来打印这些项目:

```py
for user in users:
    print(user)
```

这个`for`循环将在单独的一行上打印每个对象，但是即使这样，每个对象占用的空间也比一行所能容纳的要多。以这种方式打印确实会让事情变得好一点，但这绝不是理想的。上面的例子是一个相对简单的数据结构，但是你会用一个深度嵌套的 100 倍大小的字典做什么？

当然，你可以写一个使用[递归](https://realpython.com/python-recursion/)的函数来找到打印所有内容的方法。不幸的是，您可能会遇到一些这种方法不起作用的边缘情况。您甚至会发现自己编写了一整个函数模块，只是为了掌握数据的结构！

进入`pprint`模块！

[*Remove ads*](/account/join/)

## 使用`pprint`和工作

是一个 Python 模块，用来以漂亮的方式打印数据结构。它一直是 Python 标准库的一部分，所以没有必要单独安装它。你需要做的就是导入它的`pprint()`函数:

>>>

```py
>>> from pprint import pprint
```

然后，不要像上面的例子那样使用普通的`print(users)`方法，您可以调用您新喜欢的函数来使输出变得漂亮:

>>>

```py
>>> pprint(users)
```

这个函数打印`users`——但是以一种新的和改进的*漂亮的*方式:

>>>

```py
>>> pprint(users)
[{'address': {'city': 'Gwenborough',
 'geo': {'lat': '-37.3159', 'lng': '81.1496'},
 'street': 'Kulas Light',
 'suite': 'Apt. 556',
 'zipcode': '92998-3874'},
 'company': {'bs': 'harness real-time e-markets',
 'catchPhrase': 'Multi-layered client-server neural-net',
 'name': 'Romaguera-Crona'},
 'email': 'Sincere@april.biz',
 'id': 1,
 'name': 'Leanne Graham',
 'phone': '1-770-736-8031 x56442',
 'username': 'Bret',
 'website': 'hildegard.org'},
 {'address': {'city': 'Wisokyburgh',
 'geo': {'lat': '-43.9509', 'lng': '-34.4618'},
 'street': 'Victor Plains',
 'suite': 'Suite 879',
 'zipcode': '90566-7771'},
 'company': {'bs': 'synergize scalable supply-chains',
 'catchPhrase': 'Proactive didactic contingency',
 'name': 'Deckow-Crist'},
 'email': 'Shanna@melissa.tv',
 'id': 2,
 'name': 'Ervin Howell',
 'phone': '010-692-6593 x09125',
 'username': 'Antonette',
 'website': 'anastasia.net'},

 ...

 {'address': {'city': 'Lebsackbury',
 'geo': {'lat': '-38.2386', 'lng': '57.2232'},
 'street': 'Kattie Turnpike',
 'suite': 'Suite 198',
 'zipcode': '31428-2261'},
 'company': {'bs': 'target end-to-end models',
 'catchPhrase': 'Centralized empowering task-force',
 'name': 'Hoeger LLC'},
 'email': 'Rey.Padberg@karina.biz',
 'id': 10,
 'name': 'Clementina DuBuque',
 'phone': '024-648-3804',
 'username': 'Moriah.Stanton',
 'website': 'ambrose.net'}]
```

多漂亮啊！字典的键甚至在视觉上是缩进的！这个输出使得扫描和可视化分析数据结构变得更加简单。

**注意:**如果您自己运行代码，您将看到的输出会更长。此代码块截断输出以提高可读性。

如果你喜欢尽可能少地打字，那么你会很高兴地知道`pprint()`有一个别名，`pp()`:

>>>

```py
>>> from pprint import pp
>>> pp(users)
```

`pp()`只是`pprint()`的一个包装器，它的行为完全一样。

**注意:** Python 从[版本 3.8.0 alpha 2](https://github.com/python/cpython/tree/96831c7fcf888af187bbae8254608cccb4d6a03c) 开始就包含了这个别名。

然而，即使是默认输出也可能包含太多的信息，以至于一开始无法浏览。也许您真正想要的只是验证您正在处理的是一个普通对象的列表。为此，您需要稍微调整一下输出。

对于这些情况，有各种参数可以传递给`pprint()`来使最简洁的数据结构变得漂亮。

## 探索`pprint()`的可选参数

在本节中，您将了解到所有可用于`pprint()`的参数。有七个参数可以用来配置 Pythonic pretty 打印机。你不需要把它们都用上，有些会比其他的更有用。你会发现最有价值的可能是`depth`。

### 汇总您的数据:`depth`

最容易使用的参数之一是`depth`。如果数据结构达到或低于指定的深度，下面的 Python 命令将只打印出`users`的全部内容——当然，这一切都是为了保持美观。更深层数据结构的内容用三个点代替:

>>>

```py
>>> pprint(users, depth=1)
[{...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}]
```

现在你可以立即看到这确实是一个字典列表。为了进一步探索数据结构，您可以增加一个级别的深度，这将打印出`users`中字典的所有顶级键:

>>>

```py
>>> pprint(users, depth=2)
[{'address': {...},
 'company': {...},
 'email': 'Sincere@april.biz',
 'id': 1,
 'name': 'Leanne Graham',
 'phone': '1-770-736-8031 x56442',
 'username': 'Bret',
 'website': 'hildegard.org'},
 {'address': {...},
 'company': {...},
 'email': 'Shanna@melissa.tv',
 'id': 2,
 'name': 'Ervin Howell',
 'phone': '010-692-6593 x09125',
 'username': 'Antonette',
 'website': 'anastasia.net'},

 ...

 {'address': {...},
 'company': {...},
 'email': 'Rey.Padberg@karina.biz',
 'id': 10,
 'name': 'Clementina DuBuque',
 'phone': '024-648-3804',
 'username': 'Moriah.Stanton',
 'website': 'ambrose.net'}]
```

现在，您可以快速检查所有词典是否共享它们的顶级键。这是一个很有价值的观察，尤其是如果您的任务是开发一个像这样使用数据的应用程序。

[*Remove ads*](/account/join/)

### 给你的数据空间:`indent`

`indent`参数控制输出中每一级精美打印表示的缩进程度。默认缩进只是`1`，它转换成一个空格字符:

>>>

```py
>>> pprint(users[0], depth=1)
{'address': {...},
 'company': {...},
 'email': 'Sincere@april.biz',
 'id': 1,
 'name': 'Leanne Graham',
 'phone': '1-770-736-8031 x56442',
 'username': 'Bret',
 'website': 'hildegard.org'}

>>> pprint(users[0], depth=1, indent=4)
{   'address': {...},
 'company': {...},
 'email': 'Sincere@april.biz',
 'id': 1,
 'name': 'Leanne Graham',
 'phone': '1-770-736-8031 x56442',
 'username': 'Bret',
 'website': 'hildegard.org'}
```

`pprint()`缩进行为最重要的部分是保持所有键在视觉上对齐。应用多少缩进取决于`indent`参数和键的位置。

因为在上面的例子中没有嵌套，缩进量完全基于`indent`参数。在这两个例子中，请注意开始的花括号(`{`)是如何被算作第一个键的缩进单位的。在第一个例子中，第一个键的开始单引号紧跟在`{`之后，中间没有任何空格，因为缩进被设置为`1`。

然而，当有嵌套时，缩进应用于行内的第一个元素，然后`pprint()`保持所有后续元素与第一个对齐。因此，如果在打印`users`时将`indent`设置为`4`，第一个元素将缩进四个字符，而嵌套元素将缩进八个字符以上，因为缩进是从第一个键的末尾开始的:

>>>

```py
>>> pprint(users[0], depth=2, indent=4)
{   'address': {   'city': 'Gwenborough',
 'geo': {...},
 'street': 'Kulas Light',
 'suite': 'Apt. 556',
 'zipcode': '92998-3874'},
 'company': {   'bs': 'harness real-time e-markets',
 'catchPhrase': 'Multi-layered client-server neural-net',
 'name': 'Romaguera-Crona'},
 'email': 'Sincere@april.biz',
 'id': 1,
 'name': 'Leanne Graham',
 'phone': '1-770-736-8031 x56442',
 'username': 'Bret',
 'website': 'hildegard.org'}
```

这只是 Python 的`pprint()`中*蛮*的另一部分！

### 限制你的线路长度:`width`

默认情况下，`pprint()`每行最多只输出 80 个字符。您可以通过传入一个`width`参数来自定义这个值。`pprint()`会尽量把内容排在一行。如果一个数据结构的内容超过了这个限制，那么它将在新的一行上打印当前数据结构的每个元素:

>>>

```py
>>> pprint(users[0])
{'address': {'city': 'Gwenborough',
 'geo': {'lat': '-37.3159', 'lng': '81.1496'},
 'street': 'Kulas Light',
 'suite': 'Apt. 556',
 'zipcode': '92998-3874'},
 'company': {'bs': 'harness real-time e-markets',
 'catchPhrase': 'Multi-layered client-server neural-net',
 'name': 'Romaguera-Crona'},
 'email': 'Sincere@april.biz',
 'id': 1,
 'name': 'Leanne Graham',
 'phone': '1-770-736-8031 x56442',
 'username': 'Bret',
 'website': 'hildegard.org'}
```

当您将宽度保留为默认的 80 个字符时，`users[0]['address']['geo']`处的字典只包含一个`'lat'`和一个`'lng'`属性。这意味着，将缩进量和打印出词典所需的字符数(包括中间的空格)相加，得出的结果少于 80 个字符。由于它少于默认宽度的 80 个字符，`pprint()`将其全部放在一行中。

然而，`users[0]['company']`处的字典会超出默认宽度，所以`pprint()`将每个键放在一个新行上。字典、列表、元组和集合都是如此:

>>>

```py
>>> pprint(users[0], width=160)
{'address': {'city': 'Gwenborough', 'geo': {'lat': '-37.3159', 'lng': '81.1496'}, 'street': 'Kulas Light', 'suite': 'Apt. 556', 'zipcode': '92998-3874'},
 'company': {'bs': 'harness real-time e-markets', 'catchPhrase': 'Multi-layered client-server neural-net', 'name': 'Romaguera-Crona'},
 'email': 'Sincere@april.biz',
 'id': 1,
 'name': 'Leanne Graham',
 'phone': '1-770-736-8031 x56442',
 'username': 'Bret',
 'website': 'hildegard.org'}
```

如果将宽度设置为一个较大的值，如`160`，那么所有嵌套的字典都适合一行。您甚至可以走极端，使用像`500`这样的大值，对于本例，它在一行中打印整个字典:

>>>

```py
>>> pprint(users[0], width=500)
{'address': {'city': 'Gwenborough', 'geo': {'lat': '-37.3159', 'lng': '81.1496'}, 'street': 'Kulas Light', 'suite': 'Apt. 556', 'zipcode': '92998-3874'}, 'company': {'bs': 'harness real-time e-markets', 'catchPhrase': 'Multi-layered client-server neural-net', 'name': 'Romaguera-Crona'}, 'email': 'Sincere@april.biz', 'id': 1, 'name': 'Leanne Graham', 'phone': '1-770-736-8031 x56442', 'username': 'Bret', 'website': 'hildegard.org'}
```

在这里，您得到了将`width`设置为相对较大的值的效果。你可以反过来将`width`设置为一个较低的值，比如`1`。然而，这将产生的主要影响是确保每个数据结构将在单独的行上显示其组件。您仍然会看到排列组件的视觉缩进:

>>>

```py
>>> pprint(users[0], width=5)
{'address': {'city': 'Gwenborough',
 'geo': {'lat': '-37.3159',
 'lng': '81.1496'},
 'street': 'Kulas '
 'Light',
 'suite': 'Apt. '
 '556',
 'zipcode': '92998-3874'},
 'company': {'bs': 'harness '
 'real-time '
 'e-markets',
 'catchPhrase': 'Multi-layered '
 'client-server '
 'neural-net',
 'name': 'Romaguera-Crona'},
 'email': 'Sincere@april.biz',
 'id': 1,
 'name': 'Leanne '
 'Graham',
 'phone': '1-770-736-8031 '
 'x56442',
 'username': 'Bret',
 'website': 'hildegard.org'}
```

很难让 Python 的`pprint()`打印难看。它会尽一切努力变漂亮！

在本例中，除了学习`width`，您还将探索打印机如何拆分长文本行。注意最初为`'Multi-layered client-server neural-net'`的`users[0]["company"]["catchPhrase"]`是如何在每个空间上被分割的。打印机避免在单词中间分割这个字符串，因为那样会使它难以阅读。

[*Remove ads*](/account/join/)

### 挤压你的长序列:`compact`

您可能会认为`compact`指的是您在关于`width`的章节中探究的行为——也就是说，`compact`是让数据结构出现在一行上还是单独的行上。然而，`compact`只在一条线经过*越过*T4 时才影响输出。

**注意:** `compact`只影响序列:列表、集合、元组的输出，*不影响*字典。这是故意的，尽管不清楚为什么做出这个决定。在 [Python 第 34798 期](https://bugs.python.org/issue34798)中有一个正在进行的讨论。

如果`compact`是`True`，那么输出将换行到下一行。如果数据结构的长度超过宽度，默认行为是每个元素出现在自己的行上:

>>>

```py
>>> pprint(users, depth=1)
[{...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}]

>>> pprint(users, depth=1, width=40)
[{...},
 {...},
 {...},
 {...},
 {...},
 {...},
 {...},
 {...},
 {...},
 {...}]

>>> pprint(users, depth=1, width=40, compact=True)
[{...}, {...}, {...}, {...}, {...},
 {...}, {...}, {...}, {...}, {...}]
```

使用默认设置漂亮地打印这个列表会在一行中打印出缩略版本。将`width`限制为`40`个字符，迫使`pprint()`在单独的行上输出列表的所有元素。如果您随后设置了`compact=True`，那么列表将在 40 个字符处换行，并且比它通常看起来更紧凑。

**注意:**注意，将宽度设置为少于 7 个字符——在本例中，相当于`[{...},`的输出——似乎完全绕过了`depth`参数，并且`pprint()`最终会打印所有内容而不进行任何折叠。这已被报告为[错误#45611](https://bugs.python.org/issue45611) 。

`compact`对于包含短元素的长序列很有用，否则会占用很多行，使输出可读性更差。

### 指挥你的输出:`stream`

`stream`参数是指`pprint()`的输出。默认情况下，它和`print()`去的是同一个地方。具体到 [`sys.stdout`](https://docs.python.org/3/library/sys.html#sys.stdout) ，其实就是 Python 中的一个[文件对象](https://docs.python.org/3/glossary.html#term-file-object)。然而，您可以将它重定向到任何文件对象，就像您可以使用`print()`一样:

>>>

```py
>>> with open("output.txt", mode="w") as file_object:
...     pprint(users, stream=file_object)
```

这里你用 [`open()`](https://docs.python.org/3/library/functions.html#open) 创建一个文件对象，然后你把`pprint()`中的`stream`参数设置到那个文件对象。如果您随后打开`output.txt`文件，您应该看到您已经漂亮地打印了`users`中的所有内容。

Python 确实有自己的[日志模块](https://realpython.com/python-logging/)。然而，您也可以使用`pprint()`将漂亮的输出发送到文件中，如果您愿意的话，可以将它们作为日志。

### 防止字典排序:`sort_dicts`

虽然字典一般被认为是无序的数据结构，但是从 Python 3.6 开始，[字典是通过插入](https://docs.python.org/3.6/whatsnew/3.6.html#new-dict-implementation)排序的。

`pprint()`按字母顺序排列打印键:

>>>

```py
>>> pprint(users[0], depth=1)
{'address': {...},
 'company': {...},
 'email': 'Sincere@april.biz',
 'id': 1,
 'name': 'Leanne Graham',
 'phone': '1-770-736-8031 x56442',
 'username': 'Bret',
 'website': 'hildegard.org'}

>>> pprint(users[0], depth=1, sort_dicts=False)
{'id': 1,
 'name': 'Leanne Graham',
 'username': 'Bret',
 'email': 'Sincere@april.biz',
 'address': {...},
 'phone': '1-770-736-8031 x56442',
 'website': 'hildegard.org',
 'company': {...}}
```

除非你将`sort_dicts`设置为`False`，否则 Python 的`pprint()`会按字母顺序对键进行排序。它保持了字典输出的一致性、可读性，并且——非常漂亮！

当`pprint()`第一次实现时，字典是无序的。如果不按字母顺序排列关键字，理论上字典的关键字在每一次印刷时都会有所不同。

### 美化你的数字:`underscore_numbers`

`underscore_numbers`参数是在 [Python 3.10](https://realpython.com/python310-new-features/) 中引入的一个特性，它使得长数字更具可读性。考虑到您到目前为止使用的示例不包含任何长数字，您将需要一个新的示例来进行试验:

>>>

```py
>>> number_list = [123456789, 10000000000000]
>>> pprint(number_list, underscore_numbers=True)
[123_456_789, 10_000_000_000_000]
```

如果您尝试运行这个对`pprint()`的调用并得到一个错误，您并不孤单。截止到 2021 年 10 月，直接调用`pprint()`时，这个参数不起作用。Python 社区很快注意到了这一点，并在 2021 年 12 月 [3.10.1 bugfix 版本](https://www.python.org/dev/peps/pep-0619/#bugfix-releases)中修复了。Python 的人们关心他们漂亮的打印机！当你阅读本教程时，他们可能已经解决了这个问题。

如果当你直接调用`pprint()`时`underscore_numbers`不起作用，并且你真的想要漂亮的数字，有一个变通方法:当你创建你自己的`PrettyPrinter`对象时，这个参数应该像上面的例子一样工作。

接下来，您将讲述如何创建一个`PrettyPrinter`对象。

[*Remove ads*](/account/join/)

## 创建自定义`PrettyPrinter`对象

可以创建一个具有您定义的默认值的`PrettyPrinter`实例。一旦您有了自定义`PrettyPrinter`对象的这个新实例，您就可以通过调用`PrettyPrinter`实例上的`.pprint()`方法来使用它:

>>>

```py
>>> from pprint import PrettyPrinter
>>> custom_printer = PrettyPrinter(
...     indent=4,
...     width=100,
...     depth=2,
...     compact=True,
...     sort_dicts=False,
...     underscore_numbers=True
... )
...
>>> custom_printer.pprint(users[0])
{   'id': 1,
 'name': 'Leanne Graham',
 'username': 'Bret',
 'email': 'Sincere@april.biz',
 'address': {   'street': 'Kulas Light',
 'suite': 'Apt. 556',
 'city': 'Gwenborough',
 'zipcode': '92998-3874',
 'geo': {...}},
 'phone': '1-770-736-8031 x56442',
 'website': 'hildegard.org',
 'company': {   'name': 'Romaguera-Crona',
 'catchPhrase': 'Multi-layered client-server neural-net',
 'bs': 'harness real-time e-markets'}}
>>> number_list = [123456789, 10000000000000]
>>> custom_printer.pprint(number_list)
[123_456_789, 10_000_000_000_000]
```

使用这些命令，您可以:

*   **导入了** `PrettyPrinter`，这是一个类定义
*   用某些参数创建了该类的一个新实例
*   **打印出**中的第一个用户`users`
*   定义了一个由几个长数字组成的列表
*   **印上了`number_list`** ，这也演示了`underscore_numbers`的动作

请注意，您传递给`PrettyPrinter`的参数与默认的`pprint()`参数完全相同，除了您跳过了第一个参数。在`pprint()`中，这是你要打印的对象。

这样，您就可以有各种打印机预置—也许有些预置用于不同的流—并在需要时调用它们。

## 用`pformat()`和得到一个漂亮的字符串

如果您不想将`pprint()`的漂亮输出发送到流中，该怎么办？也许你想做一些[正则表达式](https://realpython.com/regex-python/)匹配和替换某些键。对于普通词典，您可能会发现自己想要删除括号和引号，使它们看起来更易于阅读。

无论您想对字符串预输出做什么，您都可以通过使用 [`pformat()`](https://docs.python.org/3/library/pprint.html#pprint.pformat) 来获取字符串:

>>>

```py
>>> from pprint import pformat
>>> address = pformat(users[0]["address"])
>>> chars_to_remove = ["{", "}", "'"]
>>> for char in chars_to_remove:
...     address = address.replace(char, "")
...
>>> print(address)
city: Gwenborough,
 geo: lat: -37.3159, lng: 81.1496,
 street: Kulas Light,
 suite: Apt. 556,
 zipcode: 92998-3874
```

`pformat()`是一个工具，你可以用它来连接漂亮的打印机和输出流。

另一个用例可能是，如果您正在[构建一个 API](https://realpython.com/api-integration-in-python/#rest-and-python-building-apis) ，并且想要发送一个 JSON 字符串的漂亮的字符串表示。您的最终用户可能会喜欢它！

## 处理递归数据结构

Python 的`pprint()`是递归的，这意味着它将漂亮地打印一个字典的所有内容，任何子字典的所有内容，等等。

问问自己，当递归函数遇到递归数据结构时会发生什么。假设你有字典`A`和字典`B`:

*   `A`有一个属性`.link`，指向`B`。
*   `B`有一个属性`.link`，指向`A`。

如果你想象的递归函数没有办法处理这个循环引用，它将永远不会完成打印！它会打印出`A`，然后是其子节点`B`。但是`B`小时候也有`A`，所以它会无限延续下去。

幸运的是，普通的`print()`函数和`pprint()`函数都能很好地处理这个问题:

>>>

```py
>>> A = {}
>>> B = {"link": A}
>>> A["link"] = B
>>> print(A)
{'link': {'link': {...}}}
>>> from pprint import pprint
>>> pprint(A)
{'link': {'link': <Recursion on dict with id=3032338942464>}}
```

Python 的常规`print()`只是简化了输出，`pprint()`显式地通知您递归，并添加了字典的 ID。

如果你想探究为什么这个结构是递归的，你可以学习更多关于[通过引用](https://realpython.com/python-pass-by-reference/)传递的知识。

[*Remove ads*](/account/join/)

## 结论

您已经探索了 Python 中`pprint`模块的主要用法以及使用`pprint()`和`PrettyPrinter`的一些方法。您会发现，无论何时开发处理复杂数据结构的东西时，`pprint()`都非常方便。也许你正在开发一个使用不熟悉的 API 的应用程序。也许您有一个充满深度嵌套的 JSON 文件的数据仓库。这些都是`pprint`可以派上用场的情况。

在本教程中，您学习了如何:

*   **导入** `pprint`用于您的程序
*   用 **`pprint()`** 代替常规的`print()`
*   了解所有可以用来定制精美打印输出的**参数**
*   在打印之前，将格式化的输出作为一个**字符串**获取
*   创建 **`PrettyPrinter`** 的自定义实例
*   认识**递归数据结构**以及`pprint()`如何处理它们

为了帮助您掌握函数和参数，您使用了一个代表一些用户的数据结构示例。您还探索了一些可能使用`pprint()`的情况。

恭喜你！通过使用 Python 的`pprint`模块，您现在可以更好地处理复杂数据。*****