# Python 中的定制 JSON 编码器

> 原文：<https://www.pythonforbeginners.com/basics/custom-json-encoder-in-python>

JSON 对象是在互联网上传输信息的最有效的方式之一。但是，我们不能直接将所有类型的数据转换成 JSON。在本文中，我们将讨论创建自定义 JSON 编码器的不同方法，以将 python 对象转换为 Python 中的 JSON 格式。

本文讨论如何用 Python 创建一个定制的 JSON 编码器。如果你没有使用过 JSON 文件，甚至没有使用过默认的编码器，我建议你阅读这篇关于在 Python 中使用 JSON 文件的文章。

## 使用 dump()方法将 Python 对象转换为 JSON

`dump()`方法用于将 python 对象转换成 JSON 文件。它接受 python 对象和指向 JSON 文件的文件指针，并将 JSON 格式的内容保存到 JSON 文件中，如下所示。

```py
fp=open("sample.json","w")
myDict={"Name": "Aditya", "Age": 23}
json.dump(myDict, fp)
fp.close()
```

输出:

![](img/85d0a991b6dbf73db5da49d2ce1e4bdf.png)



这里，我们将 python 字典转换成了 JSON 文件。我们可以使用`dump()`方法将几乎所有的原始数据类型转换成 JSON 文件。然而，当我们试图使用`dump()` 方法将自定义 python 对象转换为 JSON 文件时，程序运行到一个[类型错误异常](https://www.pythonforbeginners.com/basics/typeerror-in-python)，并显示消息“类型错误:type _class name_ 的对象不是 JSON 可序列化的”。

```py
class Student:
    def __init__(self, name, age):
        self.Name=name
        self.Age=age
student=Student("Aditya",23)
fp=open("sample.json","w")
json.dump(student, fp)
fp.close()
```

输出:

```py
TypeError: Object of type Student is not JSON serializable
```

在上面的例子中，我们创建了一个`Student`类。当我们将一个`Student`类的 python 对象传递给`dump()`方法时，程序会遇到 TypeError 异常。这是因为`dump()`方法不知道如何将 Student 类的对象编码成 JSON 对象。

为了避免这个错误，我们可以在`dump()`方法中使用`“default”`参数或`“cls”`参数。

### 使用“默认”参数创建定制的 JSON 编码器

当`dump()`方法获得一个不可序列化的对象作为输入时，它会引发一个 TypeError 异常。您可以使用`“default”`参数来处理这种情况。`default`参数将一个函数作为它的输入参数。该函数应该返回对象的 JSON 可序列化版本。

为了创建一个 JSON 可序列化对象，我们将使用对象的`__dict__`属性将 python 对象转换成它的字典表示。该函数将返回一个可 JSON 序列化的 dictionary 对象。因此，该对象将被转换为 JSON 文件，如下例所示。

```py
class Student:
    def __init__(self, name, age):
        self.Name=name
        self.Age=age
def SimpleEncoderFunction(obj):
    return obj.__dict__
student=Student("Aditya",23)
fp=open("samplestudent.json","w")
json.dump(student, fp,default=SimpleEncoderFunction)
fp.close()
```

输出:

![](img/85d0a991b6dbf73db5da49d2ce1e4bdf.png)



在上面的例子中，我们定义了一个名为 S `impleEncoderFunction()`的自定义编码器函数。该函数将 python 对象作为其输入参数，并返回该对象的字典表示。

每当`dump()`方法遇到不可序列化的定制 python 对象时，它就将该对象发送给 SimpleEncoderFunction。然后，`SimpleEncoderFunction()`函数将 python 对象转换成字典并返回字典。然后，`dump()`方法将字典转换成 JSON 格式。

### 使用“cls”参数创建定制的 JSON 编码器

除了使用默认参数，我们还可以使用`cls`参数通过`dump()`方法创建一个定制的 json 编码器。当我们想要将定制的 python 对象转换成 JSON 时，会用到`cls`参数。为了将自定义对象转换成 JSON，我们需要定义一个自定义 JSONEncoder 子类，并将其传递给`cls`参数。

子类应该覆盖 JSONEncoder 类的`default()`方法，并返回 python 对象的 JSON 可序列化版本。例如，考虑下面的例子。

```py
class Student:
    def __init__(self, name, age):
        self.Name=name
        self.Age=age
class SimpleEncoderClass(json.JSONEncoder):
        def default(self, obj):
            return obj.__dict__
student=Student("Aditya",23)
fp=open("samplestudent.json","w")
json.dump(student, fp,cls=SimpleEncoderClass)
fp.close()
```

输出:

![](img/85d0a991b6dbf73db5da49d2ce1e4bdf.png)



这里，`SimpleEncoderClass`是 JSONEncoder 类的一个子类。它覆盖了`default()`方法。当我们将`SimpleEncoderClass`传递给`cls`参数时，`dump()` 方法成功地从 python 对象创建了一个 JSON 文件。

在执行过程中，每当`dump()`方法遇到不可序列化的定制 python 对象时，它就使用`SimpleEncoderClass`对该对象进行编码。然后，`SimpleEncoderClass`中的`default()` 方法将 python 对象转换成字典并返回字典。然后，`dump()`方法将字典转换成 JSON 格式。

## 使用 dumps()方法将 Python 对象转换成 JSON

如果想将 python 对象转换成 JSON 字符串而不是 JSON 文件，可以使用`dumps()`方法。`dumps()`方法将 JSON 可序列化 python 对象作为其输入参数，并返回 JSON 字符串，如下例所示。

```py
myDict={"Name": "Aditya", "Age": 23}
print("The python object is:")
print(myDict)
jsonStr=json.dumps(myDict)
print("The json string is:")
print(jsonStr)
```

输出:

```py
The python object is:
{'Name': 'Aditya', 'Age': 23}
The json string is:
{"Name": "Aditya", "Age": 23} 
```

在这个例子中，我们从 python 字典中创建了一个 JSON 字符串。然而，当我们将自定义 python 对象传递给`dumps()`方法时，它会遇到如下所示的 TypeError 异常。

```py
class Student:
    def __init__(self, name, age):
        self.Name=name
        self.Age=age
student=Student("Aditya",23)
jsonStr=json.dumps(student)
print("The json string is:")
print(jsonStr)
```

输出:

```py
TypeError: Object of type Student is not JSON serializable
```

在这个例子中，您可以观察到`dumps()`方法无法从类类型`Student`的 python 对象创建 JSON 字符串。

为了避免这个错误，我们可以将一个定制的 JSON 编码器函数传递给`dumps()`方法的默认参数，如下所示。

```py
class Student:
    def __init__(self, name, age):
        self.Name=name
        self.Age=age
def SimpleEncoderFunction(obj):
    return obj.__dict__
student=Student("Aditya",23)
jsonStr=json.dumps(student,default=SimpleEncoderFunction)
print("The json string is:")
print(jsonStr)
```

输出:

```py
The json string is:
{"Name": "Aditya", "Age": 23}
```

除了默认参数，您还可以对 JSONEncoder 类的子类使用`cls`参数，如下所示。

```py
class Student:
    def __init__(self, name, age):
        self.Name=name
        self.Age=age
class SimpleEncoderClass(json.JSONEncoder):
        def default(self, obj):
            return obj.__dict__
student=Student("Aditya",23)
jsonStr=json.dumps(student,cls=SimpleEncoderClass)
print("The json string is:")
print(jsonStr)
```

输出:

```py
The json string is:
{"Name": "Aditya", "Age": 23}
```

在上面的例子中，`SimpleEncoderClass`和`SimpleEncoderFunction`使用`dumps()`方法的方式与使用`dump()`方法的方式相同。

## 使用 JSONEncoder 类将 Python 对象转换为 JSON

代替`dumps()`方法，您可以使用 JSONEncoder 类将 python 对象转换成 JSON 字符串。JSONEncoder 类构造函数具有以下语法。

```py
json.JSONEncoder(*, skipkeys=False, ensure_ascii=True, check_circular=True, allow_nan=True, sort_keys=False, indent=None, separators=None, default=None)
```

这里，

*   python 对象中的键可以是任何数据类型。但是，并非所有的数据类型都可以转换成 JSON 格式。当我们试图用不同于`str`、`int`、`float`、`bool`和`None`的数据类型对 python 对象或字典进行编码时，编码器会引发一个 TypeError 异常。`skipkeys`参数帮助我们在这种情况下处理数据。当我们将`skipkeys`设置为`True`时，编码器会跳过具有不兼容数据类型的键，而不是遇到 TypeError 异常。
*   `ensure_ascii`参数用于确保输出 JSON 文件中的所有字符都是 ASCII 字符。当`ensure_ascii`设置为`True`时，编码时会跳过所有非 ASCII 字符。如果设置为`False`，非 ASCII 字符将在编码时保存到 JSON 文件中。
*   `check_circular`参数用于确保编码器对容器类型执行循环引用检查。如果`check_circular`被设置为`False`，循环参考检查被跳过。在这种情况下，循环引用将导致程序运行到 RecursionError。
*   `allow_nan`参数用于在编码时将`NaN`和 infinity 值转换为 JSON 格式。当`allow_nan`设置为`True`时，编码器将`NaN`、`+inf`、`-inf`分别转换为 JavaScript `NaN`、`Infinity`、`-Infinity` 。当`allow_nan`设置为`False`时，编码器在将 python 对象编码为 JSON 时发现`NaN`、`+inf`或`-inf`时，会引发 [ValueError 异常](https://www.pythonforbeginners.com/exceptions/valueerror-invalid-literal-for-int-with-base-10)。
*   如果您希望 JSON 对象的键以一种排序的方式排序，那么在 JSONEncoder 类中使用`sort_keys`参数。如果将`sort_keys`参数设置为`True`，那么输出 JSON 对象的键将以字典顺序的方式排序。
*   `indent`参数用于指定 JSON 对象中的缩进。当 indent 参数设置为`None`或负整数时，JSON 对象是最紧凑的表示。当`indent`参数被设置为一个正整数值时，它在编码时创建的 JSON 对象中每层缩进那么多空格。当缩进设置为字符串时，该字符串用作缩进字符。当`indent`设置为 0 或空字符串时，每缩进一级引入一个新行。
*   默认情况下，JSON 对象的键值对由逗号分隔，键和值用冒号分隔。要在将 python 对象编码为 JSON 时为键和项指定新的分隔符，可以将包含两个字符的元组传递给`separators`参数。元组的第一个字符成为键和值的分隔符。元组的第二个元素成为不同项目的分隔符。当`indent`参数设置为`None`时，`separators`参数的默认值为 `(', ', ': ')`。否则，`separators`参数的默认值为`(',', ': ')`。为了获得最紧凑的 JSON 对象，您应该从分隔符中删除空格，并使用`(',', ':')`作为`separators`参数的输入参数。
*   当编码器在 python 对象中获得一个不可序列化的对象时，它会引发一个 TypeError 异常。您可以使用默认参数来处理这种情况。`default`参数将一个函数作为其输入参数。该函数应该返回对象的 JSON 可编码版本，或者引发 TypeError。它有助于对无法序列化的 python 对象进行编码。

要将 python 对象转换成 JSON，我们可以首先使用`JSONEncoder()`构造函数创建一个 JSONEncoder 对象。之后，我们可以调用 JSONEncoder 对象上的`encode()`方法。`encode()`方法获取 python 对象并返回一个 JSON 字符串，如下所示。

```py
myDict={"Name": "Aditya", "Age": 23}
print("The python object is:")
print(myDict)
jsonStr=json.JSONEncoder().encode(myDict)
print("The json string is:")
print(jsonStr)
```

输出:

```py
The python object is:
{'Name': 'Aditya', 'Age': 23}
The json string is:
{"Name": "Aditya", "Age": 23}
```

在上面的例子中，我们使用 JSONEncoder 类和`encode()`方法将 python 字典转换为 JSON 字符串。

当我们将一个非 json 可序列化对象传递给`encode()`方法时，程序会遇到如下所示的 TypeError 异常。

```py
class Student:
    def __init__(self, name, age):
        self.Name=name
        self.Age=age
student=Student("Aditya",23)
jsonStr=json.JSONEncoder().encode(student)
print("The json string is:")
print(jsonStr)
```

输出:

```py
TypeError: Object of type Student is not JSON serializable
```

这里，我们尝试从 Student 类的自定义 python 对象创建一个 JSON 字符串。因为 python 对象是非 json 可序列化的，所以程序运行时遇到了 TypeError 异常。

为了避免错误，我们可以将`SimpleEncoderFunction`传递给`JSONEncoder()`构造函数的`default`参数，如下所示。

```py
class Student:
    def __init__(self, name, age):
        self.Name=name
        self.Age=age
def SimpleEncoderFunction(obj):
    return obj.__dict__
student=Student("Aditya",23)
jsonStr=json.JSONEncoder(default=SimpleEncoderFunction).encode(student)
print("The json string is:")
print(jsonStr)
```

输出:

```py
The json string is:
{"Name": "Aditya", "Age": 23}
```

这里，JSONEncoder 类使用`SimpleEncoderFunction`而不是预定义的`default()` 方法将 python 对象编码成 JSON。

在上面的代码中，每当`encode()`方法遇到不可序列化的定制 python 对象时，它就将该对象发送给`SimpleEncoderFunction`。然后`SimpleEncoderFunction()`函数将 python 对象转换成字典并返回字典。然后，`encode()`方法将字典转换成 JSON 格式。

建议阅读:如果你对机器学习感兴趣，你可以在 [mlops for 初学者](https://codinginfinite.com/mlops-a-complete-guide-for-beginners/)上阅读这篇文章。您可能还会喜欢这篇关于用 Python 对混合数据类型进行聚类的[的文章。](https://codinginfinite.com/clustering-for-mixed-data-types-in-python/)

## JSON 编码器，用于将复杂的 Python 对象转换成 JSON

没有任何其他自定义对象作为其属性的 Python 对象可以很容易地使用简单的 JSON 编码器(如`SimpleEncoderFunction`或`SimpleEncoderClass`)转换成 JSON 对象。它们也可以用一个简单的 JSON 解码器转换回 python 对象，正如你在这篇关于 Python 中的[定制 JSON 解码器的文章中所看到的。](https://www.pythonforbeginners.com/basics/custom-json-decoder-in-python)

然而，简单的编码器和解码器不能很好地管理以另一个对象作为其属性的复杂 python 对象。对于复杂的 python 对象，我们可以使用上面讨论过的`SimpleEncoderClass`或`SimpleEncoderFunction`。您可以在下面的示例中观察到这一点。

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
def SimpleEncoderFunction(obj):
    return obj.__dict__
details=Details(160,60)
student=Student("Aditya",23,details)
jsonStr=json.JSONEncoder(default=SimpleEncoderFunction).encode(student)
print("The json string is:")
print(jsonStr)
```

输出:

```py
The json string is:
{"Name": "Aditya", "Age": 23, "Details": {"Height": 160, "Weight": 60}}
```

在上面的例子中，我们有一个包含`"Height"`和`"Weight"`属性的细节类。在`Student`类中，我们有`"Name"`、`"Age",`和`"Details"` 属性。

当一个`Student`类的对象作为输入被提供给`encode(`方法时，它被发送给`SimpleEncoderFunction()`。在将学生类转换成字典时，`SimpleEncoderFunction()` 遇到了`Details`类。然后，`SimpleEncoderFunction()` 函数递归调用自己，将`Details`类的对象作为输入参数。这里，该函数返回一个`Details`对象的字典表示。然后，`SimpleEncoderFunction()`返回`Student`对象的字典表示。之后，字典被`encode()`方法转换成 JSON 对象。

当我们试图将 JSON 对象转换回 python 对象时，这种类型的编码会导致错误。因此，我们还需要在 JSON 中指定 python 对象的类型。这有助于在解码 JSON 对象时避免错误。

您可以定义一个定制的 JSON 编码器函数，并与如下所示的`dump()`方法一起使用。

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
def ComplexEncoderFunction(obj):
        objDict=obj.__dict__
        typeDict={"__type__":type(obj).__name__}
        return {**objDict,**typeDict}

details=Details(160,60)
student=Student("Aditya",23,details)
fp=open("complexjson.json","w")
json.dump(student, fp,default=ComplexEncoderFunction)
fp.close()
```

输出:

![](img/85d0a991b6dbf73db5da49d2ce1e4bdf.png)



在上面的例子中，我们定义了一个`ComplexEncode` rFunction()而不是`SimpleEncoderFunction()`。在将 python 对象转换为 dictionary 时，`ComplexEncoderFunction()`添加一个名为`"__type__"`的键，以 python 对象的类名作为其关联值，然后 dictionary 通过类名转换为 JSON，帮助我们将 JSON 对象转换为具有正确类类型的自定义 python 对象。

您还可以定义一个 JSONEncoder 类的自定义子类，并与如下所示的`dump()`方法和`cls`参数一起使用。

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
class ComplexEncoderClass(json.JSONEncoder):
        def default(self, obj):
            objDict=obj.__dict__
            typeDict={"__type__":type(obj).__name__}
            return {**objDict,**typeDict}

details=Details(160,60)
student=Student("Aditya",23,details)
fp=open("complexjson.json","w")
json.dump(student, fp,cls=ComplexEncoderClass)
fp.close()
```

输出:

![](img/85d0a991b6dbf73db5da49d2ce1e4bdf.png)



您可以通过如下所示的`dumps()`方法和`default`参数使用自定义编码器功能。

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
def ComplexEncoderFunction(obj):
        objDict=obj.__dict__
        typeDict={"__type__":type(obj).__name__}
        return {**objDict,**typeDict}

details=Details(160,60)
student=Student("Aditya",23,details)
jsonStr=json.dumps(student,default=ComplexEncoderFunction)
print("The json string is:")
print(jsonStr)
```

输出:

```py
The json string is:
{"Name": "Aditya", "Age": 23, "Details": {"Height": 160, "Weight": 60, "__type__": "Details"}, "__type__": "Student"}
```

您还可以通过 dumps()方法使用 JSONEncoder 类的自定义子类，如下所示。

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
class ComplexEncoderClass(json.JSONEncoder):
        def default(self, obj):
            objDict=obj.__dict__
            typeDict={"__type__":type(obj).__name__}
            return {**objDict,**typeDict}

details=Details(160,60)
student=Student("Aditya",23,details)
jsonStr=json.dumps(student,cls=ComplexEncoderClass)
print("The json string is:")
print(jsonStr)
```

输出:

```py
The json string is:
{"Name": "Aditya", "Age": 23, "Details": {"Height": 160, "Weight": 60, "__type__": "Details"}, "__type__": "Student"}
```

除了使用自定义的 JSONEncoder 子类，您还可以使用 JSON encoder 函数和 JSON encoder 构造函数中的`default`参数来创建 JSON 编码器，并使用它将 python 对象转换为 JSON，如下例所示。

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
def ComplexEncoderFunction(obj):
        objDict=obj.__dict__
        typeDict={"__type__":type(obj).__name__}
        return {**objDict,**typeDict}

details=Details(160,60)
student=Student("Aditya",23,details)
jsonStr=json.JSONEncoder(default=ComplexEncoderFunction).encode(student)
print("The json string is:")
print(jsonStr)
```

输出

```py
The json string is:
{"Name": "Aditya", "Age": 23, "Details": {"Height": 160, "Weight": 60, "__type__": "Details"}, "__type__": "Student"}
```

## 结论

在本文中，我们讨论了创建自定义 JSON 编码器的不同方法，以将复杂的 python 对象转换为 JSON。要了解更多关于 python 编程的知识，你可以阅读这篇关于如何用 Python 创建聊天应用的文章。您可能也会喜欢这篇关于使用 Python 中的 sklearn 模块进行[线性回归的文章。](https://codinginfinite.com/linear-regression-using-sklearn-in-python/)

敬请关注更多消息灵通的文章。

快乐学习！