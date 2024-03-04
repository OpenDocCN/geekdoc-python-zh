# 在 Python 中将字符串转换为日期时间

> 原文：<https://www.pythonforbeginners.com/basics/convert-string-to-datetime-in-python>

我们通常将日期以字符串的形式保存在文本文件或电子表格中。在 python 中处理日期和时间时，我们经常需要计算两个事件之间花费的天数或时间。在这样的计算中，我们可以使用 python datetime 模块，在它的帮助下，我们可以将日期存储为 python 对象。在本文中，我们将讨论如何在 python 中将字符串转换为日期时间对象。

## Python 中的 Datetime 是什么？

Python 为我们提供了`datetime`模块来处理与时间和日期相关的数据。它定义了许多函数来计算当前时间，两个日期之间花费的时间等等。例如，您可以使用下面的`datetime.now()`函数获得当前日期和时间。

```py
import datetime

today = datetime.datetime.now()
print("The Current Date and Time is:", today) 
```

输出:

```py
The Current Date and Time is: 2021-11-22 22:34:25.828339
```

`datetime.now()`函数的输出是一个 datetime 对象，它有许多属性，如`year`、`month`、`day`、`minute`、`second`和`microsecond`。您可以按如下方式访问每个属性。

```py
import datetime

today = datetime.datetime.now()
print("The Current Year is:", today.year)
print("The Current Month is:", today.month)
print("The Current Day is:", today.day)
print("The Current Hour is:", today.hour)
print("The Current Minute is:", today.minute)
print("The Current Second is:", today.second)
print("The Current Microsecond is:", today.microsecond)
```

输出:

```py
The Current Year is: 2021
The Current Month is: 11
The Current Day is: 22
The Current Hour is: 22
The Current Minute is: 36
The Current Second is: 30
The Current Microsecond is: 972280
```

我们还可以使用`datetime()`构造函数创建一个 datetime 对象。它接受年、月和日作为其第一、第二和第三个参数，并返回一个 datetime 对象，如下所示。

```py
import datetime

day = datetime.datetime(1999, 1, 20)
print("The Date is:", day)
```

输出:

```py
The Date is: 1999-01-20 00:00:00
```

您还可以将小时、分钟、秒、微秒和时区参数传递给`datetime()`构造函数，作为 day 之后的后续参数，顺序相同。这些是可选参数，小时、分钟、秒和微秒的默认值为 0。时区的默认值为无。

## Python 中如何把字符串转换成 Datetime？

除了直接创建日期对象，我们还可以在 python 中将字符串转换为日期时间对象。我们可以使用`datetime.strptime()`方法来实现。

`datetime.strptime()`方法接受一个包含日期的字符串作为第一个输入参数，一个包含日期格式的字符串作为第二个输入参数。执行后，它返回一个 datetime 对象，如下所示。

```py
import datetime

input_string = "1999-01-20"
print("The input string is:",input_string)
date_format = "%Y-%m-%d"  # %Y for year, %m for month and %d for day
day = datetime.datetime.strptime(input_string, date_format)
print("The Date is:", day)
```

输出:

```py
The input string is: 1999-01-20
The Date is: 1999-01-20 00:00:00
```

我们还可以为输入字符串指定其他格式，并将它们转换为 datetime 对象，如下例所示。

```py
import datetime

input_string = "1999/01/20"
print("The input string is:",input_string)
date_format = "%Y/%m/%d"  # %Y for year, %m for month and %d for day
day = datetime.datetime.strptime(input_string, date_format)
print("The Date is:", day)
input_string = "20-01-1999"
print("The input string is:",input_string)
date_format = "%d-%m-%Y"  # %Y for year, %m for month and %d for day
day = datetime.datetime.strptime(input_string, date_format)
print("The Date is:", day)
input_string = "20/01/1999"
print("The input string is:",input_string)
date_format = "%d/%m/%Y"  # %Y for year, %m for month and %d for day
day = datetime.datetime.strptime(input_string, date_format)
print("The Date is:", day)
```

输出:

```py
The input string is: 1999/01/20
The Date is: 1999-01-20 00:00:00
The input string is: 20-01-1999
The Date is: 1999-01-20 00:00:00
The input string is: 20/01/1999
The Date is: 1999-01-20 00:00:00
```

## 结论

在本文中，我们讨论了 python 中的 datetime 对象。我们还讨论了如何在 python 中将字符串转换成日期时间对象。要了解更多关于字符串的知识，可以阅读这篇关于 python 中的[字符串方法的文章。您可能也会喜欢这篇关于 python](https://www.pythonforbeginners.com/basics/python-string-methods-for-string-manipulation) 中的[字符串连接的文章。](https://www.pythonforbeginners.com/concatenation/string-concatenation-and-formatting-in-python)