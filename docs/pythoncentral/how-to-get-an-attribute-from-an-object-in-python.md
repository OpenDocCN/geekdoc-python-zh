# 如何在 Python 中获得对象的属性

> 原文：<https://www.pythoncentral.io/how-to-get-an-attribute-from-an-object-in-python/>

一旦我们知道如何在 Python 中[检查一个对象是否有属性，下一步就是获取那个属性。在 Python 中，除了普通的点样式属性访问之外，还有一个内置函数`getattr`，它对于访问属性也非常有用。](https://www.pythoncentral.io/how-to-check-if-an-object-has-an-attribute-in-python/ "How to check if an object has an attribute in Python")

## **Python 的 Getattr**

内置函数`getattr(object, name[, default])`返回`object`的一个`name` d 属性的值，其中`name`必须是一个字符串。如果`object`有一个带`name`的属性，那么返回该属性的值。另一方面，如果`object`不具有带`name`的属性，则返回`default`的值，或者如果不提供`default`则提高`AttributeError`的值。

```py

>>> t = ('This', 'is', 'a', 'tuple')

>>> t.index('is')

1

>>> getattr(t, 'index')

<built-in method index of tuple object at 0x10c15e680>

>>> getattr(t, 'index')('is')

1

```

在前面的代码片段中，我们创建了一个 tuple `t`并以普通方式调用了它的方法`index`，然后以`getattr`方式调用。注意`getattr(t, 'index')`返回一个绑定到对象`t`的函数。

如果一个`object`没有带`name`的属性会怎么样？回想一下`getattr`的定义，它声明要么返回`default`的值，要么引发`AttributeError`。

```py

>>> getattr(t, 'len')

Traceback (most recent call last):

File "", line 1, in

AttributeError: 'tuple' object has no attribute 'len'

>>> getattr(t, 'len', t.count)

<built-in method count of tuple object at 0x10c15e680>

```

除了像元组、列表和类实例这样的“普通”对象，`getattr`也接受模块作为参数。因为模块也是 Python 中的对象，所以模块的属性可以像对象中的任何属性一样被检索。

```py

>>> import uuid

>>> getattr(uuid, 'UUID')

<class 'uuid.UUID'>

>>> type(getattr(uuid, 'UUID'))

<type 'type'>

>>> isinstance(getattr(uuid, 'UUID'), type)

True

>>> callable(getattr(uuid, 'UUID'))

True

>>> getattr(uuid, 'UUID')('12345678123456781234567812345678')

UUID('12345678-1234-5678-1234-567812345678')

```

### **getattr 和 Dispatcher 模式**

与访问属性的常用方法相比，`getattr`更不稳定，因为它接受一个标识属性名称的字符串，而不是在源代码中硬编码名称。所以用`getattr`做调度就很自然了。例如，如果您想编写一个程序，根据文件扩展名以不同方式转换不同类型的文件，您可以将转换逻辑编写为类中的独立函数，并根据当前文件扩展名调用它们。

```py

>>> from collections import namedtuple

>>> import os

>>> Transformer = namedtuple('Transformer', ['transform_png', 'transform_txt'])

>>> transformer = Transformer('transform .png files', 'transform .txt files')

>>> # Suppose the current directory has two files 'tmp.png' and 'tmp.txt'

>>> current_directory = os.path.dirname(os.path.abspath(__file__))

>>> for dirname, dirnames, filenames in os.walk(current_directory):

... for filename in filenames:

... path = os.path.join(dirname, filename)

... filename_prefix, ext = filename.rsplit('.', 1)

... if ext == 'png' or ext == 'txt':

... print(getattr(transformer, 'transform_{0}'.format(ext)))

...

transform .png files

transform .txt files

```

### **使用 getattr 的提示和建议**

*   尽管`getattr`在处理自省和元编程时非常方便，但它确实使代码更难阅读。所以，当你跃跃欲试要用`getattr`的时候，先静下心来想想是否真的有必要。通常情况下，您的用例可能不需要`getattr`。
*   当你确实想使用`getattr`时，写代码要比平时慢，并确保你有单元测试来支持它，因为几乎所有涉及`getatrr`的错误都是运行时错误，很难修复。通过一次正确地编写代码来节省时间比以后再调试要有效得多。