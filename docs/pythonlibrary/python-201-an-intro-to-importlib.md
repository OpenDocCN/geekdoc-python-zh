# python 201:import lib 简介

> 原文：<https://www.blog.pythonlibrary.org/2016/05/27/python-201-an-intro-to-importlib/>

Python 提供了 **importlib** 包作为其标准模块库的一部分。其目的是为 Python 的**导入**语句(以及 **__import__()** 函数)提供实现。此外，importlib 使程序员能够创建他们自己的定制对象(又名**导入器**),可以在导入过程中使用。

> 小鬼呢？
> 
> 还有一个叫做 **imp** 的模块，它为 Python 的**imp**语句背后的机制提供了一个接口。Python 3.4 中不赞成使用此模块。打算用 **importlib** 来代替它。

该模块相当复杂，因此我们将把本文的范围限制在以下主题上:

*   动态导入
*   检查一个模块是否可以导入
*   从源文件本身导入

让我们从动态导入开始吧！

* * *

### 动态导入

importlib 模块支持导入作为字符串传递给它的模块的能力。因此，让我们创建几个我们可以使用的简单模块。我们将为两个模块提供相同的接口，但是让它们打印出自己的名字，这样我们就可以区分这两个模块。创建两个不同名称的模块，如 **foo.py** 和 **bar.py** ，并在每个模块中添加以下代码:

```py
def main():
    print(__name__)

```

现在我们只需要使用 importlib 来导入它们。让我们来看看实现这一点的一些代码。确保将这段代码放在与上面创建的两个模块相同的文件夹中。

```py
# importer.py

import importlib

def dynamic_import(module):

    return importlib.import_module(module)

if __name__ == '__main__':
    module = dynamic_import('foo')
    module.main()

    module_two = dynamic_import('bar')
    module_two.main()

```

这里我们导入了方便的 importlib 模块，并创建了一个名为 **dynamic_import** 的非常简单的函数。这个函数所做的就是用我们传入的模块字符串调用 importlib 的 **import_module** 函数，并返回调用的结果。然后在我们底部的条件语句中，我们调用每个模块的 **main** 方法，它会忠实地打印出模块的名称。

您可能不会在自己的代码中经常这样做，但是偶尔您会发现自己想要导入一个模块，而此时您只有一个字符串形式的模块。importlib 模块给了我们这样做的能力。

* * *

### 模块导入检查

Python 有一种被称为 EAFP 的编码风格:请求原谅比请求许可更容易。这意味着，假设某些东西存在(比如字典中的一个键)并在我们出错时捕捉异常通常更容易。在我们上一章中你看到了这一点，我们试图导入一个模块，如果它不存在，我们就捕捉到了 **ImportError** 。如果我们想检查一个模块是否可以被导入，而不仅仅是猜测，那该怎么办？你可以用 importlib 来做！让我们来看看:

```py
import importlib.util

def check_module(module_name):
    """
    Checks if module can be imported without actually
    importing it
    """
    module_spec = importlib.util.find_spec(module_name)
    if module_spec is None:
        print('Module: {} not found'.format(module_name))
        return None
    else:
        print('Module: {} can be imported!'.format(module_name))
        return module_spec

def import_module_from_spec(module_spec):
    """
    Import the module via the passed in module specification
    Returns the newly imported module
    """
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module

if __name__ == '__main__':
    module_spec = check_module('fake_module')
    module_spec = check_module('collections')
    if module_spec:
        module = import_module_from_spec(module_spec)
        print(dir(module))

```

这里我们导入了 importlib 的一个子模块，名为 **util** 。 **check_module** 代码是我们想看的第一个魔术。在其中，我们针对传入的模块字符串调用了 **find_spec** 函数。首先我们传入一个假名，然后我们传入一个 Python 模块的真名。如果您运行这段代码，您会看到当您传入一个没有安装的模块名时， **find_spec** 函数将返回 **None** ，我们的代码将打印出没有找到该模块。如果找到了，那么我们将返回模块规范。

我们可以获取模块规范，并使用它来实际导入模块。或者您可以将字符串传递给我们在上一节中了解到的 **import_module** 函数。但是我们已经讨论过了，所以让我们来学习如何使用模块规范。看看上面代码中的 **import_module_from_spec** 函数。接受 **check_module** 返回的模块规格。然后，我们将它传递给 importlib 的 **module_from_spec** 函数，该函数返回导入模块。Python 的文档建议在导入模块后执行它，所以这就是我们接下来用 **exec_module** 函数做的事情。最后，我们返回模块并对其运行 Python 的 **dir** 以确保它是我们期望的模块。

* * *

### 从源文件导入

importlib 的 util 子模块有另一个我想介绍的巧妙技巧。你可以使用 **util** 来导入一个模块，只需要它的名字和文件路径。下面是一个非常衍生的例子，但我认为它会让你明白这一点:

```py
import importlib.util

def import_source(module_name):
    module_file_path = module_name.__file__
    module_name = module_name.__name__

    module_spec = importlib.util.spec_from_file_location(
        module_name, module_file_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    print(dir(module))

    msg = 'The {module_name} module has the following methods:' \
        ' {methods}'
    print(msg.format(module_name=module_name, 
                     methods=dir(module)))

if __name__ == '__main__':
    import logging
    import_source(logging)

```

在上面的代码中，我们实际上导入了**日志**模块，并将其传递给我们的 **import_source** 函数。一旦到了那里，我们就获取模块的实际路径及其名称。然后我们调用将这些信息传递给 util 的 **spec_from_file_location** 函数，该函数将返回模块的规范。一旦我们有了这些，我们就可以使用在上一节中使用的相同的 importlib 机制来实际导入模块。

* * *

### 包扎

此时，您应该知道如何在自己的代码中使用 importlib 和 import 挂钩。这个模块的内容比本文所介绍的要多得多，所以如果您需要编写一个自定义导入器或加载器，那么您需要花一些时间阅读文档和源代码。

* * *

### 相关阅读

*   importlib - [导入](https://docs.python.org/3/library/importlib.html)的实现
*   Python 101: [关于导入的一切](https://www.blog.pythonlibrary.org/2016/03/01/python-101-all-about-imports/)
*   Python 3 - [从 github 导入](https://www.blog.pythonlibrary.org/2016/02/02/python-3-import-from-github/)