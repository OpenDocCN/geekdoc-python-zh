# Python 中的定制 JSON 解码器

> 原文：<https://www.pythonforbeginners.com/basics/custom-json-decoder-in-python>

JSON 对象是与 web 应用程序通信的最有效的工具之一。当我们收到一个 JSON 文件时，我们需要将它转换成一个 python 对象，以便在我们的 python 程序中使用。在本文中，我们将讨论在 Python 中创建和使用定制 JSON 解码器的不同方法。

在继续这篇文章之前，如果您不知道如何使用简单的 JSON 对象，我建议您阅读这篇关于在 Python 中使用 JSON 文件的文章。

## 如何将 JSON 转换成 Python 对象？

您可以使用`load()`方法、`loads()`方法或 JSONDecoder 类将 JSON 文件或字符串转换成 python 对象。让我们逐一讨论每种方法。

### 使用 load()方法将 JSON 文件转换为 Python 对象

`load()`方法获取一个指向 JSON 文件的文件指针，并返回一个 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python)对象。例如，我们有下面的 JSON 文件。

![](img/85d0a991b6dbf73db5da49d2ce1e4bdf.png)



JSON object

当我们使用`load()`方法将这个文件转换成 python 对象时，我们得到一个 python 字典，如下例所示。

```py
import json
fp=open("simplestudent.json")
myDict=json.load(fp)
print("The python object is:")
print(myDict)
```

输出

```py
The python object is:
{'Name': 'Aditya', 'Age': 23}
```

如果你想得到一个 python 对象而不是字典，我们需要创建一个定制的 JSON 解码器。为此，我们将创建一个函数，它接受由`load()`方法返回的字典，并将其转换为 python 对象。我们将在编码 JSON 文件时将函数传递给`load()`方法中的`object_hook`参数。您可以在下面的示例中观察到这一点。

```py
import json
class Student:
    def __init__(self, name, age):
        self.Name=name
        self.Age=age
def SimpleDecoderFunction(jsonDict):
    return Student(jsonDict["Name"],jsonDict["Age"])
fp=open("simplestudent.json","r")
python_obj=json.load(fp,object_hook=SimpleDecoderFunction)
print("The python object is:")
print(python_obj)
```

输出:

```py
The python object is:
<__main__.Student object at 0x7fe1c87a36a0>
```

在上面的例子中，我们定义了一个`Student`类。我们还定义了一个`SimpleDecoderFunction()`函数。当我们在解码 JSON 对象时将`SimpleDecoderFunction()`传递给`load()` 方法时，创建的 python 字典对象首先被发送给`SimpleDecoderFunction()`。`SimpleDecoderFunction()` 获取字典并将其转换为`Student`类的 python 对象，我们将该对象作为`load()`方法的输出。

## 使用 loads()方法将 Json 字符串转换为 Python 对象

如果有 JSON 字符串而不是 JSON 文件，可以使用`loads()` 方法将其转换成 python 对象。`loads()`方法将一个 JSON 字符串作为其输入参数，并返回一个 python 字典，如下例所示。

```py
import json
jsonStr='{"Name": "Aditya", "Age": 23}'
python_obj=json.loads(jsonStr)
print("The python object is:")
print(python_obj)
```

输出:

```py
The python object is:
{'Name': 'Aditya', 'Age': 23}
```

要使用`loads()`方法将 JSON 字符串转换成 python 对象，可以使用定制的 JSON 解码器函数和如下所示的`object_hook`参数。

```py
import json
class Student:
    def __init__(self, name, age):
        self.Name=name
        self.Age=age
def SimpleDecoderFunction(jsonDict):
    return Student(jsonDict["Name"],jsonDict["Age"])
jsonStr='{"Name": "Aditya", "Age": 23}'
python_obj=json.loads(jsonStr,object_hook=SimpleDecoderFunction)
print("The python object is:")
print(python_obj)
```

输出:

```py
The python object is:
<__main__.Student object at 0x7fe1c87a17b0>
```

您可以观察到`loads()`方法的工作方式与 `load()`方法相似。唯一的区别是它从字符串而不是文件中读取 JSON 对象。

除了使用`load()` 方法和`loads()` 方法，我们还可以使用 JSONDecoder 类创建一个解码器，将 JSON 对象转换成 python 对象。

建议阅读:如果你对机器学习感兴趣，你可以在 [mlops for 初学者](https://codinginfinite.com/mlops-a-complete-guide-for-beginners/)上阅读这篇文章。您可能还会喜欢这篇关于用 Python 对混合数据类型进行聚类的[的文章。](https://codinginfinite.com/clustering-for-mixed-data-types-in-python/)

## 使用 JSONDecoder 类将 JSON 文件转换为 Python 对象

JSONDecoder 类构造函数具有以下语法

```py
class json.JSONDecoder(*, object_hook=None, parse_float=None, parse_int=None, parse_constant=None, strict=True, object_pairs_hook=None)
```

这里，

*   `object_hook`参数用于创建定制的 JSON 解码器。`object_hook`参数将一个函数作为其输入参数。使用从 JSON 解码的对象文字来调用该函数。在输出中，函数的返回值被用来代替字典。
*   `parse_float`参数用于将 JSON 中的任何浮点数转换成另一种数据类型。默认情况下，解码时用 JSON 中包含浮点数的字符串调用`float()`函数。如果我们在`parse_float`参数中指定一个函数，解码器将包含一个浮点数的字符串传递给该函数，该函数的输出在 python 对象中使用。如果您希望在加载 JSON 本身时将浮点转换为 int 或其他数据类型，可以使用这个参数。
*   `parse_int`参数用于将 JSON 中的任何整数转换成另一种数据类型。默认情况下，使用 JSON 中包含整数的字符串调用`int()`函数。如果我们在`parse_int`参数中指定一个函数，解码器将包含整数的字符串传递给该函数，函数的输出在 python 对象中使用。如果您想在加载 JSON 本身时将整数转换成浮点数或其他数据类型，可以使用这个参数。`int()`的默认`parse_int`现在通过解释器的整数字符串转换长度限制来限制整数字符串的最大长度，以帮助避免拒绝服务攻击。
*   `parse_constant`参数用于将`NaN`、 `-Infinity`和`+Infinity`从 JSON 加载到自定义 python 值中。`parse_constant`参数将一个函数作为它的输入参数。而解码器的执行，`NaN`、`-Infinity,`和`+Infinity`被传递给函数，返回值在 python 对象中使用。
*   `object_pairs_hook`是一个可选参数，它将一个函数作为其输入参数。调用该函数时，任何对象文字的解码结果都带有一个有序的对列表。使用了`object_pairs_hook`的返回值来代替字典。此功能可用于实现自定义解码器。如果`object_hook`也被定义，则`object_pairs_hook`优先。

执行后，`JSONDecoder()`构造函数返回一个 JSON 解码器。我们可以调用 JSON 解码器上的`decode()`方法从 JSON 字符串中获取 python 字典，如下所示。

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

在上面的例子中，我们首先使用`JSONDecoder()`构造函数创建一个 JSONDecoder 对象。之后，我们调用 JSONDecoder 对象上的 `decode()`方法。`decode()`对象接受一个 JSON 字符串作为其输入参数，并返回一个 Python 字典。

要将 JSON 字符串转换成自定义 python 对象，可以在`JSONDecoder()`构造函数中使用`object_hook`参数。`JSONDecoder()` 构造函数接受一个函数作为它的输入参数。该函数必须接受字典，这是解码时的正常输出，并将其转换为自定义 python 对象。例如，考虑下面的例子。

```py
import json
class Student:
    def __init__(self, name, age):
        self.Name=name
        self.Age=age
def SimpleDecoderFunction(jsonDict):
    return Student(jsonDict["Name"],jsonDict["Age"])
jsonStr='{"Name": "Aditya", "Age": 23}'
python_obj=json.JSONDecoder(object_hook=SimpleDecoderFunction).decode(jsonStr)
print("The python object is:")
print(python_obj)
```

输出:

```py
The python object is:
<__main__.Student object at 0x7fe1c87a32b0>
```

## 使用自定义解码器类将嵌套的 JSON 文件转换为 Python 对象

将平面 JSON 文件转换为 python 对象很容易，因为 JSON 对象中的所有值在转换为字典时都是原始数据类型。然而，解码嵌套的 JSON 对象给了我们嵌套的字典。

![](img/85d0a991b6dbf73db5da49d2ce1e4bdf.png)



如果我们将上面的字符串转换成 JSON，我们将得到如下所示的嵌套字典。

```py
import json
jsonStr='{"__type__": "Student","Name": "Aditya", "Age": 23, "Details": {"__type__": "Details","Height": 160, "Weight": 60}}'
python_obj=json.JSONDecoder().decode(jsonStr)
print("The python object is:")
print(python_obj)
```

输出:

```py
The python object is:
{'__type__': 'Student', 'Name': 'Aditya', 'Age': 23, 'Details': {'__type__': 'Details', 'Height': 160, 'Weight': 60}}
```

为了将嵌套的 JSON 文件转换成 python 对象，JSON 中应该有一个键-值对来决定我们想要创建的 python 对象的类型。如果 JSON 对象包含要创建的 python 对象的类型，我们可以定义一个自定义函数，该函数获取从 JSON 对象加载的字典，并将其转换为 python 对象。然后，我们将把函数传递给`load()` 方法中的`object_hook`参数。此后，`load()`方法将返回一个定制的 python 对象，而不是一个字典。您可以在下面的示例中观察到这一点。

```py
class Student:
    def __init__(self, name, age,details):
        self.Name=name
        self.Age=age
        self.Details=details
class Details:
    def __init__(self, height, weight):
        self.Height=height
        self.Weight=weight
def ComplexDecoderFunction(jsonDict):
    if '__type__' in jsonDict and jsonDict['__type__'] == 'Student':
        return Student(jsonDict['Name'], jsonDict['Age'], jsonDict['Details'])
    if '__type__' in jsonDict and jsonDict['__type__'] == 'Details':
        return Details(jsonDict['Height'], jsonDict['Weight'])

fp=open("complexstudent.json")
python_obj=json.load(fp,object_hook=ComplexDecoderFunction)
print("The python object is:")
print(python_obj)
fp.close()
```

输出:

```py
The python object is:
<__main__.Student object at 0x7fe1c87a2d70>
```

在上面的例子中，我们定义了一个属性为`Height`和`Weight`的`Details`类。我们还用属性`Name`、`Age`和`Details`定义了`Student`类。

为了将输入嵌套字典转换成 python 对象，我们定义了一个定制的 JSON 解码器函数`ComplexDecoderFunction()`。输入 json 对象具有属性`__type__` 来指定对象可以转换成的 python 对象的类。将复杂的 python 对象编码成 json 的过程将在这篇关于 Python 中的[定制 JSON 编码器的文章中讨论。](https://www.pythonforbeginners.com/basics/custom-json-encoder-in-python)

`load()` 方法将外部字典和内部字典传递给`ComplexDecoderFunction()`。该函数使用`__type__`属性检查字典必须转换到的类，并返回适当类型的 python 对象。然后，`load()`方法返回完整的 python 对象。

如果想从 json 字符串而不是文件中获取 python 对象，可以使用`loads()` 方法而不是 `load()`方法，如下例所示。

```py
class Student:
    def __init__(self, name, age,details):
        self.Name=name
        self.Age=age
        self.Details=details
class Details:
    def __init__(self, height, weight):
        self.Height=height
        self.Weight=weight
def ComplexDecoderFunction(jsonDict):
    if '__type__' in jsonDict and jsonDict['__type__'] == 'Student':
        return Student(jsonDict['Name'], jsonDict['Age'], jsonDict['Details'])
    if '__type__' in jsonDict and jsonDict['__type__'] == 'Details':
        return Details(jsonDict['Height'], jsonDict['Weight'])

jsonStr='{"__type__": "Student","Name": "Aditya", "Age": 23, "Details": {"__type__": "Details","Height": 160, "Weight": 60}}'
python_obj=json.loads(jsonStr,object_hook=ComplexDecoderFunction)
print("The python object is:")
print(python_obj)
```

输出:

```py
The python object is:
<__main__.Student object at 0x7fe1c87a1f90>
```

您还可以为嵌套的 json 字符串创建一个定制的解码器，以使用如下所示的`JSONDecoder()`构造函数创建一个 python 对象。

```py
class Student:
    def __init__(self, name, age,details):
        self.Name=name
        self.Age=age
        self.Details=details
class Details:
    def __init__(self, height, weight):
        self.Height=height
        self.Weight=weight
def ComplexDecoderFunction(jsonDict):
    if '__type__' in jsonDict and jsonDict['__type__'] == 'Student':
        return Student(jsonDict['Name'], jsonDict['Age'], jsonDict['Details'])
    if '__type__' in jsonDict and jsonDict['__type__'] == 'Details':
        return Details(jsonDict['Height'], jsonDict['Weight'])

jsonStr='{"__type__": "Student","Name": "Aditya", "Age": 23, "Details": {"__type__": "Details","Height": 160, "Weight": 60}}'
python_obj=json.JSONDecoder(object_hook=ComplexDecoderFunction).decode(jsonStr)
print("The python object is:")
print(python_obj)
```

输出:

```py
The python object is:
<__main__.Student object at 0x7fe1c87a31f0>
```

## 结论

在本文中，我们讨论了用 python 创建定制 json 解码器的不同方法。要了解更多关于 python 编程的知识，你可以阅读这篇关于如何用 Python 创建聊天应用的文章。您可能也会喜欢这篇关于使用 Python 中的 sklearn 模块进行[线性回归的文章。](https://codinginfinite.com/linear-regression-using-sklearn-in-python/)