# 定义函数时使用 Python 可选参数

> 原文：<https://realpython.com/python-optional-arguments/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**用可选参数定义 Python 函数**](/courses/defining-python-functions-with-optional-arguments/)

定义自己的函数是编写干净有效代码的基本技能。在本教程中，您将探索用于定义带可选参数的 Python 函数的技术。当您掌握 Python 可选参数时，您将能够定义更强大、更灵活的函数。

在本教程中，您将学习:

*   **参数**和**参数**有什么区别
*   如何定义带有**可选参数**和**默认参数值**的函数
*   如何使用 **`args`** 和 **`kwargs`** 定义函数
*   如何处理关于可选参数的**错误消息**

为了从本教程中获得最大收益，您需要熟悉用必需参数定义函数的[。](https://realpython.com/defining-your-own-python-function/)

**免费奖励:** 并学习 Python 3 的基础知识，如使用数据类型、字典、列表和 Python 函数。

## 在 Python 中创建重用代码的函数

你可以把一个函数想象成一个运行在另一个程序或另一个函数中的迷你程序。主程序调用迷你程序，并发送迷你程序运行时需要的信息。当这个函数完成所有的动作后，它可能会将一些数据发送回调用它的主程序。

函数的主要目的是允许您在需要时重用其中的代码，如果需要的话可以使用不同的输入。

当您使用函数时，您正在扩展您的 Python 词汇。这可以让你以更清晰、更简洁的方式表达问题的解决方案。

在 Python 中，按照惯例，应该用小写字母命名函数，并用下划线分隔单词，比如`do_something()`。这些约定在 [PEP 8](https://www.python.org/dev/peps/pep-0008/) 中有描述，这是 Python 的风格指南。当你调用它的时候，你需要在函数名后面加上括号。因为函数代表动作，所以最好用动词开始函数名，这样代码可读性更好。

[*Remove ads*](/account/join/)

### 定义没有输入参数的函数

在本教程中，您将使用一个基本程序的例子，该程序创建并维护一个购物清单，当您准备去超市时，[将它打印出来。](https://realpython.com/python-print/)

从创建购物清单开始:

```py
shopping_list = {
    "Bread": 1,
    "Milk": 2,
    "Chocolate": 1,
    "Butter": 1,
    "Coffee": 1,
}
```

您正在使用一个[字典](https://realpython.com/python-dicts/)来存储商品名称作为键，以及您需要购买的每件商品的数量作为值。您可以定义一个函数来显示购物清单:

```py
# optional_params.py

shopping_list = {
    "Bread": 1,
    "Milk": 2,
    "Chocolate": 1,
    "Butter": 1,
    "Coffee": 1,
}

def show_list():
    for item_name, quantity in shopping_list.items():
        print(f"{quantity}x {item_name}")

show_list()
```

当您运行这个脚本时，您将得到购物清单的打印输出:

```py
$ python optional_params.py
1x Bread
2x Milk
1x Chocolate
1x Butter
1x Coffee
```

您定义的函数没有输入参数，因为函数签名中的括号为空。签名是函数定义中的第一行:

```py
def show_list():
```

在这个例子中你不需要任何输入参数，因为字典`shopping_list`是一个全局变量**。这意味着可以从程序中的任何地方访问它，包括从函数定义中。这被称为**全球范围**。你可以在[Python Scope&LEGB 规则:解析代码中的名称](https://realpython.com/python-scope-legb-rule/)中阅读更多关于作用域的内容。*

*以这种方式使用全局变量不是一种好的做法。这可能导致几个函数对同一个数据结构进行更改，从而导致难以发现的错误。在本教程的后面部分，当您将字典作为参数传递给函数时，您将看到如何改进这一点。

在下一节中，您将定义一个具有输入参数的函数。

### 用必需的输入参数定义函数

现在，您可以初始化一个空字典，并编写一个允许您向购物列表添加项目的函数，而不是直接在代码中编写购物列表:

```py
# optional_params.py

shopping_list = {}

# ...

def add_item(item_name, quantity):
    if item_name in shopping_list.keys():
        shopping_list[item_name] += quantity
    else:
        shopping_list[item_name] = quantity

add_item("Bread", 1)
print(shopping_list)
```

函数[遍历字典的键](https://realpython.com/iterate-through-dictionary-python/)，如果键存在，数量增加。如果该项目不是其中一个键，则创建该键并为其分配一个值`1`。您可以[运行这个脚本](https://realpython.com/run-python-scripts/)来显示打印出来的字典:

```py
$ python optional_params.py
{'Bread': 1}
```

您已经在函数签名中包含了两个**参数**:

1.  `item_name`
2.  `quantity`

参数还没有任何值。函数定义中的代码使用了参数名。当您调用该函数时，您在括号内传递**个参数**，每个参数一个。参数是传递给函数的值。

参数和实参之间的区别经常被忽略。这是一个微妙但重要的区别。有时，您可能会发现参数被称为**形式参数**，参数被称为**实际参数**。

调用`add_item()`时输入的参数是必需的参数。如果您尝试在没有参数的情况下调用函数，您将会得到一个错误:

```py
# optional_params.py

shopping_list = {}

def add_item(item_name, quantity):
    if item_name in shopping_list.keys():
        shopping_list[item_name] += quantity
    else:
        shopping_list[item_name] = quantity

add_item() print(shopping_list)
```

[回溯](https://realpython.com/python-traceback/)将给出一个`TypeError`，说明参数是必需的:

```py
$ python optional_params.py
Traceback (most recent call last):
  File "optional_params.py", line 11, in <module>
    add_item()
TypeError: add_item() missing 2 required positional arguments: 'item_name' and 'quantity'
```

在本教程的后面部分，您将看到更多与使用错误数量的参数或以错误的顺序使用参数相关的错误消息。

[*Remove ads*](/account/join/)

## 使用带有默认值的 Python 可选参数

在本节中，您将学习如何定义一个接受可选参数的函数。带有可选参数的函数在使用方式上更加灵活。您可以使用或不使用参数来调用函数，如果函数调用中没有参数，则使用默认值。

### 分配给输入参数的默认值

您可以修改函数`add_item()`，使参数`quantity`具有默认值:

```py
# optional_params.py

shopping_list = {}

def add_item(item_name, quantity=1):
    if item_name in shopping_list.keys():
        shopping_list[item_name] += quantity
    else:
        shopping_list[item_name] = quantity

add_item("Bread") add_item("Milk", 2) print(shopping_list)
```

在函数签名中，您已经将默认值`1`添加到参数`quantity`中。这并不意味着`quantity`的值永远是`1`。如果在调用函数时传递了一个对应于`quantity`的参数，那么这个参数将被用作参数的值。但是，如果您没有传递任何参数，那么将使用默认值。

带默认值的参数后面不能跟常规参数。在本教程的后面，您将了解到更多关于定义参数的顺序。

函数`add_item()`现在有一个必需参数和一个可选参数。在上面的代码示例中，您调用了两次`add_item()`。您的第一个函数调用只有一个参数，它对应于所需的参数`item_name`。这种情况下，`quantity`默认为`1`。您的第二个函数调用有两个参数，所以在这种情况下不使用默认值。您可以在下面看到它的输出:

```py
$ python optional_params.py
{'Bread': 1, 'Milk': 2}
```

您还可以将必需的和可选的参数作为关键字参数传递给函数。关键字参数也可以称为命名参数:

```py
add_item(item_name="Milk", quantity=2)
```

现在，您可以重新访问您在本教程中定义的第一个函数，并重构它，使它也接受默认参数:

```py
def show_list(include_quantities=True):
    for item_name, quantity in shopping_list.items():
        if include_quantities:
            print(f"{quantity}x {item_name}")
        else:
            print(item_name)
```

现在当你使用`show_list()`时，你可以不带输入参数调用它或者传递一个[布尔值](https://realpython.com/python-boolean/)作为**标志参数**。如果在调用该函数时没有传递任何参数，那么将通过显示每件商品的名称和数量来显示购物列表。如果您在调用该函数时将`True`作为参数传递，该函数将显示相同的输出。但是，如果您使用`show_list(False)`，则只会显示项目名称。

在标志的值显著改变函数行为的情况下，应该避免使用标志。一个功能应该只负责一件事。如果你想用一个标志把函数推到另一个路径，你可以考虑写一个单独的函数。

### 常见默认参数值

在上面的例子中，一种情况下使用了整数`1`作为默认值，另一种情况下使用了布尔值`True`。这些是函数定义中常见的默认值。但是，您应该为默认值使用的数据类型取决于您正在定义的函数以及您希望如何使用该函数。

整数`0`和`1`是当参数值需要是整数时使用的常见默认值。这是因为`0`和`1`通常是有用的后备值。在您之前编写的`add_item()`函数中，将一个新物品的数量设置为`1`是最合理的选择。

然而，如果你习惯在去超市的时候买两样东西，那么将默认值设置为`2`可能更适合你。

当输入参数需要是一个[字符串](https://realpython.com/python-strings/)时，一个常用的缺省值是空字符串(`""`)。这将分配一个数据类型为 string 的值，但不会放入任何额外的字符。您可以修改`add_item()`,使两个参数都是可选的:

```py
def add_item(item_name="", quantity=1):
    if item_name in shopping_list.keys():
        shopping_list[item_name] += quantity
    else:
        shopping_list[item_name] = quantity
```

您已经修改了该函数，使两个参数都有默认值，因此可以在没有输入参数的情况下调用该函数:

```py
add_item()
```

这行代码将向`shopping_list`字典中添加一个条目，以一个空字符串作为键，值为`1`。在调用函数时检查是否传递了参数并相应地运行一些代码是相当常见的。为此，您可以更改上述函数:

```py
def add_item(item_name="", quantity=1):
    if not item_name:
        quantity = 0
    if item_name in shopping_list.keys():
        shopping_list[item_name] += quantity
    else:
        shopping_list[item_name] = quantity
```

在这个版本中，如果没有项目被传递给该函数，该函数将数量设置为`0`。空字符串有一个 **falsy** 值，这意味着`bool("")`返回`False`，而任何其他字符串将有一个 **truthy** 值。当一个`if`关键字后跟一个 the 或 falsy 值时， [`if`语句](https://realpython.com/python-conditional-statements/)会将这些值解释为`True`或`False`。你可以在 [Python 布尔值:用真值优化你的代码](https://realpython.com/python-boolean/)中阅读更多关于真值和假值的内容。

因此，您可以在`if`语句中直接使用该变量来检查是否使用了可选参数。

另一个常用的默认值是 [`None`](https://realpython.com/null-in-python/) 。这是 Python 表示空值的方式，尽管它实际上是一个表示空值的对象。在下一节中，您将看到一个例子，说明什么时候`None`是一个有用的默认值。

[*Remove ads*](/account/join/)

### 不应用作默认参数的数据类型

在上面的例子中，您已经使用了整数和字符串作为默认值，而`None`是另一个常见的默认值。这些不是唯一可以用作默认值的[数据类型](https://realpython.com/python-data-types/)。但是，并不是所有的数据类型都应该使用。

在这一节中，您将看到为什么**可变的**数据类型不应该在函数定义中用作默认值。可变对象是其值可以改变的对象，如列表或字典。你可以在 Python 的[不变性](https://realpython.com/courses/immutability-python/)和 [Python 的官方文档](https://docs.python.org/3/library/stdtypes.html#immutable-sequence-types)中找到更多关于可变和不可变数据类型的信息。

您可以将包含项目名称和数量的字典作为输入参数添加到您之前定义的函数中。首先，您可以将所有参数都设置为必需参数:

```py
def add_item(item_name, quantity, shopping_list):
    if item_name in shopping_list.keys():
        shopping_list[item_name] += quantity
    else:
        shopping_list[item_name] = quantity

    return shopping_list
```

现在可以在调用函数时将`shopping_list`传递给它。这使得函数更加自包含，因为它不依赖于调用函数的[范围](https://realpython.com/python-namespaces-scope/)中的变量`shopping_list`。这一变化也使该功能更加灵活，因为您可以将它用于不同的输入词典。

您还添加了 [`return`语句](https://realpython.com/python-return-statement/)来返回修改后的字典。从技术上讲，这一行在这个阶段是不需要的，因为字典是一种可变的数据类型，因此该函数将改变主模块中存在的字典的状态。然而，当您使这个参数可选时，您将需要`return`语句，所以最好现在就包含它。

要调用该函数，您需要将该函数返回的数据赋给一个变量:

```py
shopping_list = add_item("Coffee", 2, shopping_list)
```

您还可以向本教程中定义的第一个函数`show_list()`添加一个`shopping_list`参数。现在，您的程序中可以有多个购物清单，并使用相同的函数来添加商品和显示购物清单:

```py
# optional_params.py

hardware_store_list = {} supermarket_list = {} 
def show_list(shopping_list, include_quantities=True):
    print()
    for item_name, quantity in shopping_list.items():
        if include_quantities:
            print(f"{quantity}x {item_name}")
        else:
            print(item_name)

def add_item(item_name, quantity, shopping_list):
    if item_name in shopping_list.keys():
        shopping_list[item_name] += quantity
    else:
        shopping_list[item_name] = quantity

    return shopping_list

hardware_store_list = add_item("Nails", 1, hardware_store_list) hardware_store_list = add_item("Screwdriver", 1, hardware_store_list) hardware_store_list = add_item("Glue", 3, hardware_store_list) 
supermarket_list = add_item("Bread", 1, supermarket_list) supermarket_list = add_item("Milk", 2, supermarket_list) 
show_list(hardware_store_list) show_list(supermarket_list)
```

您可以在下面看到这段代码的输出。首先显示从五金店购买的物品清单。输出的第二部分显示了超市需要的商品:

```py
$ python optional_params.py

1x Nails
1x Screwdriver
3x Glue

1x Bread
2x Milk
```

现在您将为`add_item()`中的参数`shopping_list`添加一个默认值，这样如果没有字典传递给函数，那么将使用一个空字典。最诱人的选择是让默认值成为一个空字典。您很快就会明白为什么这不是一个好主意，但是您现在可以尝试这个选项:

```py
# optional_params.py

def show_list(shopping_list, include_quantities=True):
    print()
    for item_name, quantity in shopping_list.items():
        if include_quantities:
            print(f"{quantity}x {item_name}")
        else:
            print(item_name)

def add_item(item_name, quantity, shopping_list={}):
    if item_name in shopping_list.keys():
        shopping_list[item_name] += quantity
    else:
        shopping_list[item_name] = quantity

    return shopping_list

clothes_shop_list = add_item("Shirt", 3) show_list(clothes_shop_list)
```

当您运行这个脚本时，您将得到下面的输出，显示服装店需要的商品，这可能会给人这样的印象，即这段代码按预期工作:

```py
$ python optional_params.py

3x Shirt
```

然而，这段代码有一个严重的缺陷，可能会导致意想不到的错误结果。您可以使用`add_item()`添加一个电子商店所需物品的新购物清单，其中没有与`shopping_list`相对应的参数。这会导致使用默认值，您希望这会创建一个新的空字典:

```py
# optional_params.py

def show_list(shopping_list, include_quantities=True):
    print()
    for item_name, quantity in shopping_list.items():
        if include_quantities:
            print(f"{quantity}x {item_name}")
        else:
            print(item_name)

def add_item(item_name, quantity, shopping_list={}):
    if item_name in shopping_list.keys():
        shopping_list[item_name] += quantity
    else:
        shopping_list[item_name] = quantity

    return shopping_list

clothes_shop_list = add_item("Shirt", 3)
electronics_store_list = add_item("USB cable", 1) show_list(clothes_shop_list)
show_list(electronics_store_list)
```

当您查看以下代码的输出时，您会发现问题所在:

```py
$ python optional_params.py

3x Shirt
1x USB cable

3x Shirt
1x USB cable
```

两个购物清单是相同的，即使您每次调用函数时都将来自`add_item()`的输出分配给不同的变量。出现这个问题是因为字典是一种可变的数据类型。

在定义函数时，您将一个空字典指定为参数`shopping_list`的默认值。第一次调用这个函数时，这个字典是空的。但是，由于字典是可变类型，当您为字典赋值时，默认字典不再是空的。

当您第二次调用该函数并且再次需要`shopping_list`的默认值时，默认字典不再像第一次调用该函数时那样为空。因为调用的是同一个函数，所以使用的是存储在内存中的同一个默认字典。

不可变数据类型不会发生这种行为。这个问题的解决方案是使用另一个默认值，比如`None`，然后在没有传递可选参数时在函数内创建一个空字典:

```py
# optional_params.py

def show_list(shopping_list, include_quantities=True):
    print()
    for item_name, quantity in shopping_list.items():
        if include_quantities:
            print(f"{quantity}x {item_name}")
        else:
            print(item_name)

def add_item(item_name, quantity, shopping_list=None):
 if shopping_list is None: shopping_list = {}    if item_name in shopping_list.keys():
        shopping_list[item_name] += quantity
    else:
        shopping_list[item_name] = quantity

    return shopping_list

clothes_shop_list = add_item("Shirt", 3)
electronics_store_list = add_item("USB cable", 1)
show_list(clothes_shop_list)
show_list(electronics_store_list)
```

您可以使用`if`语句检查字典是否已经作为参数传递。你不应该依赖于`None`的虚假性质，而应该明确检查参数是否为`None`。如果传递了另一个为 false 的参数，依赖于`None`将被视为 false 值这一事实可能会导致问题。

现在，当您再次运行您的脚本时，您将获得正确的输出，因为每次您使用带有默认值`shopping_list`的函数时，都会创建一个新的字典:

```py
$ python optional_params.py

3x Shirt

1x USB cable
```

在定义带有可选参数的函数时，应该始终避免使用可变数据类型作为默认值。

[*Remove ads*](/account/join/)

### 与输入参数相关的错误消息

您将遇到的最常见的错误消息之一是，当您调用一个需要参数的函数时，却没有在函数调用中传递参数:

```py
# optional_params.py

# ...

def add_item(item_name, quantity, shopping_list=None):
    if shopping_list is None:
        shopping_list = {}
    if item_name in shopping_list.keys():
        shopping_list[item_name] += quantity
    else:
        shopping_list[item_name] = quantity

    return shopping_list

add_item()
```

这里，您调用了`add_item()`，而没有传递必需的参数`item_name`和`quantity`。每当缺少一个必需的参数时，就会得到一个`TypeError`:

```py
$ python optional_params.py
  File "optional_params.py", line 15
    add_item()
TypeError: add_item() missing 2 required positional arguments: 'item_name' and 'quantity'
```

在这种情况下，错误消息很有用。错误信息并不总是像这个一样有用。然而，在学习用必需和可选参数定义函数时，缺少必需参数并不是您遇到的唯一错误消息。

当函数定义中的参数都没有默认值时，可以任意方式对参数进行排序。当所有参数都有默认值时，同样适用。但是，当一些参数有默认值而另一些没有默认值时，定义参数的顺序很重要。

您可以尝试在`add_item()`的定义中交换参数的顺序，有默认值和没有默认值:

```py
# optional_params.py

# ...

def add_item(shopping_list=None, item_name, quantity):
    if shopping_list is None:
        shopping_list = {}
    if item_name in shopping_list.keys():
        shopping_list[item_name] += quantity
    else:
        shopping_list[item_name] = quantity

    return shopping_list
```

运行这段代码时您将得到的错误消息相当清楚地解释了这条规则:

```py
$ python optional_params.py
  File "optional_params.py", line 5
    def add_item(shopping_list=None, item_name, quantity):
                                                ^
SyntaxError: non-default argument follows default argument
```

没有默认值的参数必须始终位于有默认值的参数之前。在上面的例子中，`item_name`和`quantity`必须总是被赋值作为参数。首先使用默认值放置参数会使函数调用不明确。前两个必需的参数后面可以跟一个可选的第三个参数。

## 使用`args`和`kwargs`

您需要了解另外两种类型的 Python 可选参数。在本教程的前几节中，您已经学习了如何创建带有可选参数的函数。如果需要更多可选参数，可以在定义函数时使用默认值创建更多参数。

但是，可以定义一个接受任意数量可选参数的函数。您甚至可以定义接受任意数量的关键字参数的函数。关键字参数是具有与之相关联的关键字和值的参数，您将在接下来的部分中了解到这一点。

要定义输入参数和关键字数量可变的函数，您需要了解`args`和`kwargs`。在本教程中，我们将看看关于这些 Python 可选参数你需要知道的最重要的几点。如果你想了解更多，可以在[进一步探索`args`和`kwargs`](https://realpython.com/python-kwargs-and-args/) 。

### 接受任意数量参数的函数

在定义一个接受任意数量参数的函数之前，您需要熟悉[解包操作符](https://docs.python.org/3/tutorial/controlflow.html#unpacking-argument-lists)。您可以从如下列表开始:

>>>

```py
>>> some_items = ["Coffee", "Tea", "Cake", "Bread"]
```

变量`some_items`指向一个列表，而这个列表又包含四个条目。如果您使用`some_items`作为`print()`的参数，那么您将传递一个变量给`print()`:

>>>

```py
>>> print(some_items)
['Coffee', 'Tea', 'Cake', 'Bread']
```

如你所料，显示列表。然而，如果您必须在`print()`的括号内使用`*some_items`，您将得到不同的结果:

>>>

```py
>>> print(*some_items)
Coffee Tea Cake Bread
```

这一次，`print()`显示四个单独的字符串，而不是列表。这相当于编写以下内容:

>>>

```py
>>> print("Coffee", "Tea", "Cake", "Bread")
Coffee Tea Cake Bread
```

当星号或星形符号(`*`)紧接在一个序列之前使用时，例如`some_items`，它会将该序列解包为其单独的组成部分。当一个序列(比如一个列表)被解包时，它的项被提取出来并作为单独的对象对待。

您可能已经注意到,`print()`可以接受任意数量的参数。在上面的例子中，您已经使用了一个输入参数和四个输入参数。也可以使用带空括号的`print()`，它会打印一个空行。

现在，您已经准备好定义自己的函数，接受可变数量的输入参数。暂时可以简化`add_items()`只接受购物清单中想要的商品名称。您将为每个项目设置数量为`1`。然后，在下一节中，您将回到将数量作为输入参数的一部分。

使用`args`包含可变数量输入参数的函数签名如下所示:

```py
def add_items(shopping_list, *args):
```

您经常会看到函数签名使用名称`args`来表示这种类型的可选参数。然而，这只是一个参数名。`args`这个名字没什么特别的。正是前面的`*`赋予了这个参数特殊的属性，您将在下面读到。通常，最好使用最符合您需求的参数名称，以使代码更具可读性，如下例所示:

```py
# optional_params.py

shopping_list = {}

def show_list(shopping_list, include_quantities=True):
    print()
    for item_name, quantity in shopping_list.items():
        if include_quantities:
            print(f"{quantity}x {item_name}")
        else:
            print(item_name)

def add_items(shopping_list, *item_names):
 for item_name in item_names: shopping_list[item_name] = 1 
    return shopping_list

shopping_list = add_items(shopping_list, "Coffee", "Tea", "Cake", "Bread") show_list(shopping_list)
```

调用`add_items()`时的第一个参数是必需的参数。在第一个参数之后，函数可以接受任意数量的附加参数。在本例中，您在调用函数时添加了四个额外的参数。下面是上面代码的输出:

```py
$ python optional_params.py

1x Coffee
1x Tea
1x Cake
1x Bread
```

通过查看一个简化的示例，您可以理解`item_names`参数发生了什么:

>>>

```py
>>> def add_items_demo(*item_names):
...     print(type(item_names))
...     print(item_names)
...
>>> add_items_demo("Coffee", "Tea", "Cake", "Bread")
<class 'tuple'>
('Coffee', 'Tea', 'Cake', 'Bread')
```

当显示数据类型时，可以看到`item_names`是一个[元组](https://realpython.com/python-lists-tuples/#python-tuples)。因此，所有附加参数都被指定为元组`item_names`中的项目。然后，您可以在函数定义中使用这个元组，就像您在上面的`add_items()`的主定义中所做的那样，其中您使用一个`for`循环来遍历元组`item_names`。

这与在函数调用中将元组作为参数传递是不同的。使用`*args`允许您更灵活地使用函数，因为您可以添加任意多的参数，而不需要在函数调用中将它们放在元组中。

如果在调用函数时没有添加任何额外的参数，那么元组将是空的:

>>>

```py
>>> add_items_demo()
<class 'tuple'>
()
```

当您将`args`添加到一个函数定义中时，您通常会将它们添加在所有必需的和可选的参数之后。您可以在`args`后面有仅关键字的参数，但是对于本教程，您可以假设`args`通常会添加在所有其他参数之后，除了`kwargs`，您将在下一节中了解到。

[*Remove ads*](/account/join/)

### 接受任意数量关键字参数的函数

定义带参数的函数时，可以选择使用非关键字参数或关键字参数来调用函数:

>>>

```py
>>> def test_arguments(a, b):
...     print(a)
...     print(b)
...
>>> test_arguments("first argument", "second argument")
first argument
second argument
>>> test_arguments(a="first argument", b="second argument")
first argument
second argument
```

在第一个函数调用中，参数通过位置传递，而在第二个函数调用中，参数通过关键字传递。如果使用关键字参数，则不再需要按照定义的顺序输入参数:

>>>

```py
>>> test_arguments(b="second argument", a="first argument")
first argument
second argument
```

您可以通过声明[仅位置参数](https://realpython.com/defining-your-own-python-function/#positional-only-arguments)或[仅关键字参数](https://realpython.com/defining-your-own-python-function/#keyword-only-arguments)来改变这种默认行为。

定义函数时，可以使用`kwargs`包含任意数量的可选关键字参数，它代表关键字参数。函数签名如下所示:

```py
def add_items(shopping_list, **kwargs):
```

参数名`kwargs`前面有两个星号(`**`)。双星号或星号的操作与您之前使用的单星号类似，用于从序列中解包项目。双星用于从**地图**中打开物品。映射是一种将成对的值作为项目的数据类型，例如字典。

参数名`kwargs`经常在函数定义中使用，但是参数可以有任何其他名称，只要它前面有`**`操作符。您现在可以重写`add_items()`,使其接受任意数量的关键字参数:

```py
# optional_params.py

shopping_list = {}

def show_list(shopping_list, include_quantities=True):
    print()
    for item_name, quantity in shopping_list.items():
        if include_quantities:
            print(f"{quantity}x {item_name}")
        else:
            print(item_name)

def add_items(shopping_list, **things_to_buy):
 for item_name, quantity in things_to_buy.items(): shopping_list[item_name] = quantity 
    return shopping_list

shopping_list = add_items(shopping_list, coffee=1, tea=2, cake=1, bread=3) show_list(shopping_list)
```

这段代码的输出显示了字典`shopping_list`中的商品，显示了您希望购买的所有四种商品及其各自的数量。在调用函数时，您将此信息作为关键字参数包括在内:

```py
$ python optional_params.py

1x coffee
2x tea
1x cake
3x bread
```

前面已经了解到`args`是一个 tuple，函数调用中使用的可选非关键字实参作为条目存储在 tuple 中。可选的关键字参数存储在字典中，关键字参数作为键值对存储在该字典中:

>>>

```py
>>> def add_items_demo(**things_to_buy):
...     print(type(things_to_buy))
...     print(things_to_buy)
...
>>> add_items_demo(coffee=1, tea=2, cake=1, bread=3)
<class 'dict'>
{'coffee': 1, 'tea': 2, 'cake': 1, 'bread': 3}
```

要了解更多关于`args`和`kwargs`的信息，你可以阅读 [Python args 和 kwargs:demystemized](https://realpython.com/python-kwargs-and-args/)，你会发现更多关于函数中关键字和非关键字参数的细节，以及在定义你自己的 Python 函数的[中参数的使用顺序。](https://realpython.com/defining-your-own-python-function/#keyword-only-arguments)

## 结论

定义自己的函数来创建自包含的子例程是编写代码时的关键构建块之一。最有用和最强大的功能是那些执行一个明确的任务并且你可以灵活使用的功能。使用可选参数是实现这一点的关键技术。

**在本教程中，您已经学习了:**

*   **参数**和**参数**有什么区别
*   如何定义带有**可选参数**和**默认参数值**的函数
*   如何使用 **`args`** 和 **`kwargs`** 定义函数
*   如何处理关于可选参数的**错误消息**

对可选参数的良好理解也将有助于您使用标准库中和其他第三方模块中的函数。显示这些函数的文档将向您展示函数签名，从中您将能够识别哪些参数是必需的，哪些是可选的，哪些是`args`或`kwargs`。

然而，您在本教程中学到的主要技能是定义您自己的函数。现在，您可以开始编写带有必需和可选参数以及可变数量的非关键字和关键字参数的函数。掌握这些技能将帮助您将 Python 编码提升到一个新的水平。

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**用可选参数定义 Python 函数**](/courses/defining-python-functions-with-optional-arguments/)*********