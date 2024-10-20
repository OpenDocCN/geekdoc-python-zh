# python NONE——关于 NONE 对象你需要知道的一切

> 原文：<https://www.askpython.com/python/oops/python-none>

嘿，伙计们！在本文中，我们将重点关注 **Python NONE 关键字**。

* * *

## Python NONE 对象的工作

`Python NONE`是 Python 世界中的一个对象——面向对象编程。在 PHP、JAVA 等其他编程语言中，你可以把它看作“空”值。

**NONE 对象的数据类型为‘NoneType’**，因此，它不能被视为某些原始数据类型的值或布尔值。

因此，我们可以将 NONE 赋值给任何正在使用的变量。让我们用例子来理解没有人的需要。

考虑一个登录表单，用 Python 这样的后端语言连接到数据库。如果我们希望检查是否已经建立了到指定数据库的连接，我们可以为数据库连接对象指定 NONE，并验证连接是否安全。

现在，让我们理解 Python NONE 对象的结构。

* * *

## Python NONE 对象的语法

NONE 对象不遵循普通数据类型的注意事项。

**语法:**

```py
variable = NONE

```

此外，通过将变量赋值为 NONE，它描述了特定变量表示无值或空值。

现在让我们通过下面的例子来实现 Python NONE 对象。

* * *

## 通过示例实现无

让我们看看下面的例子。这里，我们给变量“var”赋值为 NONE。

**示例 1:将 NONE 对象赋给 Python 变量**

```py
var = None
print("Value of var: ",var)

```

当我们试图打印存储在变量中的值时，它显示了下面的输出。从而清楚地表明 NONE 对象表示可以被认为是空值的 NONE 值。

**输出:**

```py
Value of var:  None

```

在下面的例子中，我们试图检查 Python NONE 对象是否表示一个等价的布尔值。

**示例:针对 NONE 对象的布尔检查**

```py
var = None
print("Boolean Check on NONE keyword:\n")
if var:
  print("TRUE")
else:
  print("FALSE")

```

如下所示，结果是错误的。因此，这个例子清楚地表明，Python NONE 对象与布尔或其他原始类型的对象值不同。

**输出:**

```py
Boolean Check on NONE keyword:
FALSE

```

现在，让我们尝试将原始类型和非类型值加入到 Python 数据结构中，例如[集合](https://www.askpython.com/python/set/python-set)、[列表](https://www.askpython.com/python/list/python-list)等。

**示例:Python NONE with Set**

当我们将其他原始类型值和 NONE 一起传递给数据结构(如集合、列表等)时，我们会发现 NONE 值在打印它们时返回“NONE”值。

```py
var_set = {10,None,30,40,50}
for x in var_set:
    print(x)

```

**输出:**

```py
40
50
10
30
None

```

**示例:Python NONE with List**

```py
var_lst = [10,None,30,40,50]
for x in var_lst:
    print(str(x))

```

**输出:**

```py
10
None
30
40
50

```

* * *

## 结论

到此，我们就结束了这个话题。如果您遇到任何疑问，请随时在下面发表评论。

快乐学习！！

* * *

## 参考

*   [Python 无对象—文档](https://docs.python.org/3/c-api/none.html)