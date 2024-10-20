# 使用 Python 和 Pint 包进行单位转换

> 原文：<https://www.blog.pythonlibrary.org/2021/09/01/unit-conversion-pint/>

你需要经常测量吗？从一种计量单位转换到另一种计量单位怎么样？有一个名为 [Pint](https://pint.readthedocs.io/en/stable/) 的 Python 包，使得处理数量变得容易。Pint 允许你在数值和数量之间进行算术运算。你可以在 Pint 的 [GitHub 项目](https://github.com/hgrecco/pint/blob/master/pint/default_en.txt)中看到许多不同的单元类型。

让我们从学习如何安装 Pint 开始！

## 装置

您可以像这样使用 **pip** 安装 **Pint** :

```py
python3 -m pip install pint
```

如果您是一个 **conda** 用户，那么您可能希望使用以下命令:

```py
conda install -c conda-forge pint
```

现在你已经安装了 Pint，你可以开始学习如何使用它了！

## 品脱入门

Pint 最酷的特性之一是你可以用它从一种单位类型转换到另一种。例如，您可能想将某种英制单位转换为公制单位。

一个常见的用例是将英里转换为公里。打开您的 Python REPL(或空闲),尝试以下代码:

```py
>>> from pint import UnitRegistry
>>> ureg = UnitRegistry()
>>> distance = 5 * ureg.mile
>>> distance
<Quantity(5, 'mile')>
>>> distance.to("kilometer")
<Quantity(8.04672, 'kilometer')>
```

这里你创建一个名为**距离**的数量对象。您将其值设置为 5 英里。然后，要将其转换为千米，您可以调用距离的 **to()** 方法，并传入您想要的新数量名称。结果是 5 英里换算成 8.04672 公里。

您也可以在同一系统内将数量转换为不同的单位类型。例如，如果需要，您可以将千米转换为厘米:

```py
>>> from pint import UnitRegistry 
>>> ureg = UnitRegistry()
>>> distance_in_km = 5 * ureg.kilometer
>>> distance_in_km
<Quantity(5, 'kilometer')>
>>> distance_in_cm = distance_in_km.to("centimeter")
>>> distance_in_cm
<Quantity(500000.0, 'centimeter')>
```

## Pint 解析字符串

Pint 的一个很酷的特性是你可以用字符串来指定数量。这意味着你可以这样做:

```py
>>> my_quantity = ureg.Quantity
>>> my_quantity(2.54, 'centimeter')
<Quantity(2.54, 'centimeter')>
```

或者你可以简化它，甚至更简单:

```py
>>> my_quantity = ureg.Quantity
>>> my_quantity('2.54in')
<Quantity(2.54, 'inch')>
```

现在您已经在不同的单位类型之间进行了转换，您已经准备好学习使用 Pint 进行字符串格式化了。

## 对 Pint 使用字符串格式

Pint 支持使用 Python 的**进行格式化。format()** 并通过使用 f 字符串。这里有一个来自[品脱教程](https://pint.readthedocs.io/en/stable/tutorial.html)的例子:

```py
>>> ureg = ureg.Quantity
>>> accel = 1.3 * ureg['meter/second**2']
>>> print(f'The str is {accel}')
The str is 1.3 meter / second ** 2
```

当对 f 字符串求值时，Quantity 对象被转换成更易于阅读的格式。

Pint 通过扩展 Python 的格式化功能走得更远。以下是他们定制的“漂亮印刷品”的一个例子:

```py
>>> ureg = ureg.Quantity
>>> accel = 1.3 * ureg['meter/second**2']
>>> # Pretty print
>>> 'The pretty representation is {:P}'.format(accel)
'The pretty representation is 1.3 meter/second²'
```

Pint 还支持 Jupyter 笔记本的 LaTeX 和 HTML 定制打印。

## 包扎

Pint 是一个非常好的 Python 包。虽然本教程没有涉及到它，Pint 允许您设置您的区域设置，以便单元名称与您的语言相匹配。如果您经常处理需要在单位类型之间进行转换的量(如厘米到毫米或英寸到厘米)，这可能正是您需要的，可以使您的编码工作更容易。

## 更简洁的 Python 包

想了解其他优秀的第三方 Python 包吗？查看以下文章:

*   arrow—[Python 的新日期/时间包](https://www.blog.pythonlibrary.org/2014/08/05/arrow-a-new-date-time-package-for-python/)

*   [对 sh 包的简单介绍](https://www.blog.pythonlibrary.org/2016/01/20/a-brief-intro-to-the-sh-package/)

*   [Python 图像库/ Pillow 简介](https://www.blog.pythonlibrary.org/2016/10/07/an-intro-to-the-python-imaging-library-pillow/)