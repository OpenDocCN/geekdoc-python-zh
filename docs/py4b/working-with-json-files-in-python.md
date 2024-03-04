# 在 Python 中使用 JSON 文件

> 原文：<https://www.pythonforbeginners.com/basics/working-with-json-files-in-python>

JSON 文件是两个 web 应用程序之间通信最常用的数据格式之一。在本文中，我们将讨论什么是 JSON 对象，以及如何在 Python 中处理 JSON 文件。

## JSON 对象是什么？

JSON 是 JavaScript 对象符号的首字母缩写。它是一种基于文本的标准格式，使用键值对以结构化的方式表示数据。它最常用于在 web 应用程序之间传输数据。JSON 文件的扩展名为`.json`。

下面是一个 JSON 对象的例子。

```py
{"Name": "Aditya", "Age":23, "Height":181}
```

### 什么是嵌套的 JSON 对象？

嵌套的 JSON 对象是包含另一个 JSON 对象作为一个或多个键的关联值的 JSON 对象。我们还可以使用嵌套的 JSON 对象来传输数据。例如，考虑下面的 JSON 对象。

```py
{ 
    "coffee": {
        "region": [
            {"id":1, "name": "John Doe"},
            {"id":2, "name": "Don Josh"}
        ],
        "country": {"id":2, "company": "ACME"}
    }, 
    "brewing": {
        "region": [
            {"id":1, "name": "John Doe"},
            {"id":2, "name": "Don Josh"}
        ],
        "country": {"id":2, "company": "ACME"}
    }
} 
```

上面的 JSON 对象是嵌套的 JSON 对象。您可以观察以下内容。

*   在外部对象中，我们有两个键，即`“coffee”`和`“brewing”.`
*   `“coffee”`和`“brewing”`键包含其他 JSON 对象作为它们的值。因此，给定的 JSON 对象是一个嵌套的 JSON 对象。
*   在`“coffee”` 和`“brewing”`里面，我们有两个按键，分别是`“region”`和`“country”`。 `“country”`包含另一个 JSON 对象作为它的键，而`“region”`包含一个 JSON 对象列表。

因此，一个嵌套的 JSON 对象可以包含另一个 JSON 对象或一组 JSON 对象。

## 定义 JSON 对象的语法

正如您在上面两个例子中看到的，JSON 对象具有以下语法。

*   数据以类似于 python 字典的方式呈现在键值对中。
*   JSON 字符串中的键和值由冒号`(:)`分隔。
*   JSON 对象中的每个键值对由逗号`(,)`分隔。
*   JSON 对象中的所有数据都用花括号 `({ })`括起来。
*   字符串值和键用双引号`(“ ”)`括起来。
*   数组用方括号 `([ ])`括起来。
*   数组中的值用逗号分隔。数组中的值可以是 JSON 对象、数组或任何允许的数据类型的文字。
*   JSON 对象中的键是字符串数据类型。另一方面，关联值可以是允许的数据类型之一。允许的数据类型有字符串、数字、对象、数组、布尔值或空值。

## 在 Python 中使用 JSON 文件

Python 为我们提供了 JSON 模块来处理 Python 中的 JSON 字符串和文件。现在让我们讨论如何将 python 对象转换成 JSON 对象，反之亦然。

## Python 对象到 JSON 文件

我们可以使用 dump()方法将 python 对象转换成 JSON 文件。

### dump()方法

`dump()`方法的语法如下。

```py
json.dump(python_obj, fp, *, skipkeys=False, ensure_ascii=True, check_circular=True, allow_nan=True, cls=None, indent=None, separators=None, default=None, sort_keys=False, **kw)
```

这里，

*   `python_obj`参数接受一个需要转换成 JSON 文件的 python 对象。该对象可以是数字、字符串、字典、列表或自定义 python 对象。
*   `fp`参数将一个文件指针作为它的输入参数。在写模式下打开一个扩展名为`.json`的文件后，可以将它传递给`fp`参数。执行后，`python_obj`的内容以 JSON 格式保存到`fp`指向的文件中。
*   python 对象中的键可以是任何数据类型。但是，并非所有的数据类型都可以转换成 JSON 格式。当我们试图从一个 python 对象或字典创建一个 JSON 文件，其键的数据类型不是`str`、`int`、`float`、`bool`和`None`时，`dump()`方法会引发一个[类型错误异常](https://www.pythonforbeginners.com/basics/typeerror-in-python)。`skipkeys`参数帮助我们在这种情况下处理数据。当我们将`skipkeys`设置为`True`时，`dump()`方法会跳过具有不兼容数据类型的键，而不是遇到 TypeError 异常。
*   `ensure_ascii`参数用于确保输出 JSON 文件中的所有字符都是 ASCII 字符。当`ensure_ascii`设置为`True`时，跳过`python_obj`中所有非 ASCII 字符。如果设置为`False`，非 ASCII 字符将按原样保存到 JSON 文件中。
*   `check_circular`参数用于确保`dump()`方法对容器类型执行循环引用检查。如果`check_circular`被设置为`False`，循环参考检查被跳过。在这种情况下，循环引用将导致程序运行到 RecursionError 异常。
*   `allow_nan`参数用于将`NaN`和 infinity 值转换为 JSON 格式。当`allow_nan`设置为`True`时，`dump()`方法将`NaN`、`+inf`和`-inf` 分别转换为 JavaScript `NaN`、`Infinity`、`-Infinity`。当`allow_nan`设置为`False`时，`dump()`方法在`python_obj`中找到`NaN`、`+inf`或`-inf` 时，会引发 [ValueError 异常](https://www.pythonforbeginners.com/exceptions/valueerror-invalid-literal-for-int-with-base-10)。
*   当我们想要将定制的 python 对象转换成 JSON 时，使用`cls`参数。为了将自定义对象转换成 JSON，我们需要定义一个自定义的`JSONEncoder`子类，并将其传递给`cls`参数。
*   `indent`参数用于指定 JSON 对象中的缩进。当`indent`参数被设置为`None`或负整数时，JSON 对象是最紧凑的表示。当`indent`参数被设置为一个正整数值时，它在从`dump()`方法创建的 JSON 对象中每层缩进那么多空格。当`indent`被设置为字符串时，该字符串被用作缩进字符。当`indent`设置为 0 或空字符串时，每缩进一级引入一个新行。
*   默认情况下，JSON 对象的键值对由逗号分隔，键和值用冒号分隔。要为键和项指定新的分隔符，可以将包含两个字符的元组传递给`separators`参数。元组的第一个字符成为键和值的分隔符。元组的第二个元素成为不同项目的分隔符。当`indent`参数设置为`None`时，`separators`参数的默认值为`(', ', ': ')`。否则，`separators`参数的默认值为`(',', ': ')`。为了获得最紧凑的 JSON 对象，您应该从分隔符中删除空格，并使用`(',', ':')`作为`separators`参数的输入参数。
*   当`dump()` 方法在`python_obj`参数中获得一个不可序列化的对象时，它会引发一个 TypeError 异常。您可以使用`default`参数来处理这种情况。`default`参数将一个函数作为其输入参数。该函数应该返回对象的 JSON 可编码版本，或者引发 TypeError。
*   如果您希望 JSON 对象的键以一种排序的方式排序，那么在`dump()` 方法中使用`sort_keys`参数。如果`sort_keys`参数被设置为`True`，则输出 JSON 对象的键以词典的方式排序。

执行后，`dump()`方法将 JSON 文件保存到给定的文件指针。您可以在下面的示例中观察到这一点。

```py
import json
myStr="Aditya Raj"
fp=open("fromstring.json","w")
json.dump(myStr, fp)
fp.close()
```

输出:

![](img/85d0a991b6dbf73db5da49d2ce1e4bdf.png)



在上面的例子中，我们已经将字符串`"Aditya Raj"` 转换为名为`fromstring.json`的 JSON 文件。为此，我们首先使用`open()`函数以写模式打开`fromstring.json`文件。`open()`函数将文件名和文字`"w"`作为它的输入参数。执行后，它以写模式打开文件并返回一个文件指针。我们将文件指针和输入字符串传递给`dump()`方法。执行后，`dump()`方法将 JSON 对象保存在文件中。

最后，我们使用`close()`方法关闭文件。如果不关闭文件，写入文件的任何数据都不会被保存。因此，这是重要的一步。

您还可以使用 Python 中的`dumps()`方法将列表和字典等对象转换成 JSON 文件，如下所示。

```py
import json
myDict={"Name": "Aditya", "Age": 23}
fp=open("fromdict.json","w")
json.dump(myDict, fp)
fp.close()
```

输出:

![](img/85d0a991b6dbf73db5da49d2ce1e4bdf.png)



在本文中，我们已经用 python 将一个 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python)转换成了一个 JSON 文件。类似地，您也可以使用`dump()` 方法将列表转换成 JSON 文件。

要将定制 python 对象转换成 JSON，可以阅读这篇关于 Python 中的[定制 JSON 编码器的文章。](https://www.pythonforbeginners.com/basics/custom-json-encoder-in-python)

## Python 对象到 JSON 字符串

### dumps()方法

dumps()方法用于将 python 对象转换为 JSON 格式的字符串。它具有以下语法。

```py
json.dumps(python_obj, *, skipkeys=False, ensure_ascii=True, check_circular=True, allow_nan=True, cls=None, indent=None, separators=None, default=None, sort_keys=False, **kw)
```

`dumps()`方法中的所有参数与 `dump()` 方法中的各个参数含义相同。唯一的区别是，`dump()`方法将 JSON 对象保存到一个文件中，而`dumps()`方法在执行后返回一个 JSON 格式的字符串。您可以在下面的示例中观察到这一点。

```py
import json
myStr="Aditya Raj"
print("The input string is:")
print(myStr)
jsonStr=json.dumps(myStr)
print("The JSON string is:")
print(jsonStr)
```

输出:

```py
The input string is:
Aditya Raj
The JSON string is:
"Aditya Raj"
```

您还可以使用 Python 中的`dumps()`方法将列表和字典等对象转换为 JSON 字符串，如下所示。

```py
import json
myDict={"Name": "Aditya", "Age": 23}
print("The dictionary is:")
print(myDict)
jsonStr=json.dumps(myDict)
print("The JSON string is:")
print(jsonStr)
```

输出

```py
The dictionary is:
{'Name': 'Aditya', 'Age': 23}
The JSON string is:
{"Name": "Aditya", "Age": 23}
```

## 使用 JSONEncoder 类将 Python 对象转换为 JSON 字符串

JSONEncoder 类用于创建默认和定制的 JSON 编码器，以便将 python 对象转换为 JSON 格式。执行时，`JSONEncoder()`构造函数返回一个 JSONEncoder 对象。

我们可以调用 JSONEncoder 对象上的`encode()`方法来从 python 对象创建 JSON 字符串。当在 JSONEncoder 对象上调用`encode()` 方法时，该方法将 python 对象作为其输入参数，并返回 python 对象的 JSON 表示。您可以在下面的示例中观察到这一点。

```py
import json
myStr="Aditya Raj"
print("The input string is:")
print(myStr)
jsonStr=json.JSONEncoder().encode(myStr)
print("The JSON string is:")
print(jsonStr)
```

输出:

```py
The input string is:
Aditya Raj
The JSON string is:
"Aditya Raj"
```

在这个例子中，我们首先使用`JSONEncoder()`构造函数创建了一个 JSONEncoder 对象。然后，我们使用`encode()` 方法将 python 字符串转换成 JSON 字符串。

除了原始数据类型，还可以将列表和字典等容器对象转换成 JSON 格式，如下所示。

```py
import json
myDict={"Name": "Aditya", "Age": 23}
print("The dictionary is:")
print(myDict)
jsonStr=json.JSONEncoder().encode(myDict)
print("The JSON string is:")
print(jsonStr)
```

输出:

```py
The dictionary is:
{'Name': 'Aditya', 'Age': 23}
The JSON string is:
{"Name": "Aditya", "Age": 23}
```

建议阅读:如果你对机器学习感兴趣，你可以在 [mlops for 初学者](https://codinginfinite.com/mlops-a-complete-guide-for-beginners/)上阅读这篇文章。您可能还会喜欢这篇关于用 Python 对混合数据类型进行聚类的[的文章。](https://codinginfinite.com/clustering-for-mixed-data-types-in-python/)

## 将 JSON 文件加载到 Python 对象

我们可以使用`load()`方法将 JSON 文件加载到 python 对象中。

### load()方法

`load()`方法的语法如下。

```py
json.load(fp, *, cls=None, object_hook=None, parse_float=None, parse_int=None, parse_constant=None, object_pairs_hook=None, **kw)
```

*   `fp`参数是指向包含 JSON 文件的 file 对象的文件指针。
*   当我们想要将 JSON 转换成一个定制的 python 对象时，就要用到`cls`参数。为了将 JSON 转换成自定义对象，我们需要定义一个自定义 JSONDecoder 子类，并将其传递给`cls`参数。
*   `object_hook`参数用于创建定制的 JSON 解码器。`object_hook`参数将一个函数作为其输入参数。使用从 JSON 解码的对象文字来调用该函数。在输出中，函数的返回值被用来代替字典。
*   `parse_float`参数用于将 JSON 中的任何浮点数转换成另一种数据类型。默认情况下，使用 JSON 中包含浮点数的字符串调用`float()`函数。如果我们在`parse_float`参数中指定一个函数，`load()` 方法将包含一个浮点数的字符串传递给该函数，该函数的输出在 python 对象中使用。如果您希望在加载 JSON 本身时将浮点转换为 int 或其他数据类型，可以使用这个参数。
*   `parse_int`参数用于将 JSON 中的任何整数转换成另一种数据类型。默认情况下，使用 JSON 中包含整数的字符串调用`int()`函数。如果我们在`parse_int`参数中指定一个函数，`load()`方法将包含整数的字符串传递给该函数，函数的输出在 python 对象中使用。如果您想在加载 JSON 本身时将整数转换成浮点数或其他数据类型，可以使用这个参数。`int()`的默认`parse_int`现在通过解释器的整数字符串转换长度限制来限制整数字符串的最大长度，以帮助避免拒绝服务攻击。
*   `parse_constant`参数用于将`NaN`、`-Infinity`和`+Infinity`从 JSON 加载到自定义 python 值中。`parse_constant`参数将一个函数作为它的输入参数。而`load()`函数的执行，`NaN`、`-Infinity`、`+Infinity`被传递给函数，返回值在 python 对象中使用。
*   `object_pairs_hook`是一个可选参数，它将一个函数作为其输入参数。调用该函数时，任何对象文字的解码结果都带有一个有序的对列表。使用了`object_pairs_hook`的返回值来代替字典。此功能可用于实现自定义解码器。如果`object_hook`也被定义，则`object_pairs_hook`优先。

执行后，`load()`方法返回一个 python 对象。例如，考虑下面的 JSON 文件。

![](img/85d0a991b6dbf73db5da49d2ce1e4bdf.png)



当我们使用`load()`方法将上面的 JSON 文件转换成 python 对象时，我们将得到一个 python 字典。

您可以使用如下所示的`load()`方法将 JSON 文件转换成 python 对象。

```py
import json
fp=open("simplestudent.json","r")
myDict=json.load(fp)
print("The python object is:")
print(myDict)
fp.close()
```

输出:

```py
The python object is:
{'Name': 'Aditya', 'Age': 23}
```

## 将 JSON 字符串转换为 Python 对象

要将 JSON 字符串转换成 python 对象，我们可以使用 loads()方法或 JSONDecoder 类。

### loads()方法

`loads()`方法用于将 JSON 字符串加载到 python 对象中。它具有以下语法。

```py
json.loads(json_string, *, cls=None, object_hook=None, parse_float=None, parse_int=None, parse_constant=None, object_pairs_hook=None, **kw)
```

这里，`json_string`参数表示必须转换成 python 对象的 JSON 字符串。`loads()`方法中的所有其他参数与`load()` 方法中的参数相同。执行后，`loads()`方法返回一个 python 对象，如下所示。

```py
import json
jsonStr='{"Name": "Aditya", "Age": 23}'
print("The JSON string is:")
print(jsonStr)
myDict=json.loads(jsonStr)
print("The python object is:")
print(myDict)
```

输出:

```py
The JSON string is:
{"Name": "Aditya", "Age": 23}
The python object is:
{'Name': 'Aditya', 'Age': 23}
```

在这个例子中，您可以观察到我们已经使用`loads()`方法将 JSON 字符串转换为 python 字典。

## 使用 JSONDecoder 类将 JSON 字符串转换为 Python 字典

JSONDecoder 类用于在 Python 中创建定制的 JSON 解码器。为了使用 JSONDecoder 类将 JSON 字符串转换成 python 对象，我们将首先执行`JSONDecoder()` 构造函数。JSONDecoder 构造函数在执行后返回一个 JSONDecoder 对象。

我们将调用 JSONDecoder 对象上的`decode()`方法，从 JSON 字符串创建一个 python 对象。`decode()`方法接受一个 JSON 字符串并返回一个 python 对象，如下例所示。

```py
import json
jsonStr='{"Name": "Aditya", "Age": 23}'
print("The JSON string is:")
print(jsonStr)
myDict=json.JSONDecoder().decode(jsonStr)
print("The python object is:")
print(myDict)
```

输出:

```py
The JSON string is:
{"Name": "Aditya", "Age": 23}
The python object is:
{'Name': 'Aditya', 'Age': 23}
```

默认情况下，`load()`方法、`loads()`方法和`decode()`函数返回一个 python 字典。要将 JSON 对象直接转换成定制 python 对象，可以阅读这篇关于 Python 中[定制 JSON 解码器的文章。](https://www.pythonforbeginners.com/basics/custom-json-decoder-in-python)

## 为什么要用 JSON 文件配合 Python 进行数据传输？

*   JSON 格式在语法上类似于原始的 python 对象。因此，很容易将 python 对象转换成 JSON 并通过网络发送。为了发送定制的 python 对象，我们可以定义编码器和解码器，并轻松地以 JSON 格式传输数据。
*   JSON 数据是文本格式的。因此，我们可以将它发送到任何应用程序。此外，它可以由任何编程语言处理，因为所有编程语言都支持文本数据。
*   JSON 格式是非常轻量级的。由于它很小，可以很容易地通过 HTTP 和 HTTPS 发送。
*   JSON 易于阅读，使用键值对进行结构化，不像 XML 等其他格式那样有很多结束或开始标记。
*   几乎每种主要语言都有处理 JSON 数据的专用库。因此，即使您在不同的团队中使用不同的编程语言，您的软件模块也可以使用 JSON 轻松地相互通信。

## 结论

在本文中，我们讨论了在 Python 中使用 JSON 文件。要了解更多关于 python 编程的知识，你可以阅读这篇关于如何用 Python 创建聊天应用的文章。您可能也会喜欢这篇关于使用 Python 中的 sklearn 模块进行[线性回归的文章。](https://codinginfinite.com/linear-regression-using-sklearn-in-python/)

请继续关注更多内容丰富的文章。

快乐学习！