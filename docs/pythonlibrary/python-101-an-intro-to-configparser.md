# python 101:config parser 简介

> 原文：<https://www.blog.pythonlibrary.org/2013/10/25/python-101-an-intro-to-configparser/>

用户和程序员都使用配置文件。它们通常用于存储应用程序的设置，甚至是操作系统的设置。Python 的核心库包括一个名为 **ConfigParser** 的模块，您可以使用它来创建配置文件并与之交互。在本文中，我们将花几分钟时间了解它是如何工作的。

**注意:本文中的例子使用的是 Python 2。在 Python 3 中，ConfigParser 被重命名为 configparser。**

### 创建配置文件

用 ConfigParser 创建配置文件非常简单。让我们创建一些代码来演示:

```py

import ConfigParser

#----------------------------------------------------------------------
def createConfig(path):
    """
    Create a config file
    """
    config = ConfigParser.ConfigParser()
    config.add_section("Settings")
    config.set("Settings", "font", "Courier")
    config.set("Settings", "font_size", "10")
    config.set("Settings", "font_style", "Normal")
    config.set("Settings", "font_info",
               "You are using %(font)s at %(font_size)s pt")

    with open(path, "wb") as config_file:
        config.write(config_file)

#----------------------------------------------------------------------
if __name__ == "__main__":
    path = "settings.ini"
    createConfig(path)

```

上面的代码将创建一个配置文件，其中一个部分标记为 **Settings** ，包含四个选项:font、font_size、font_style 和 font_info。还要注意，当我们将配置写入磁盘时，我们使用“wb”标志以二进制模式写入。这不是必需的，但是官方文档示例就是这样做的。您也可以使用普通的“w”标志，它仍然可以工作。我在文档中找不到任何东西来解释其中的原因。

### 如何读取、更新和删除选项

现在我们正在阅读学习如何阅读配置文件，更新它的选项，甚至如何删除选项。在这种情况下，通过实际编写一些代码来学习更容易！只需将下面的函数添加到您上面编写的代码中。

```py

import ConfigParser
import os

#----------------------------------------------------------------------
def crudConfig(path):
    """
    Create, read, update, delete config
    """
    if not os.path.exists(path):
        createConfig(path)

    config = ConfigParser.ConfigParser()
    config.read(path)

    # read some values from the config
    font = config.get("Settings", "font")
    font_size = config.get("Settings", "font_size")

    # change a value in the config
    config.set("Settings", "font_size", "12")

    # delete a value from the config
    config.remove_option("Settings", "font_style")

    # write changes back to the config file
    with open(path, "wb") as config_file:
        config.write(config_file)

#----------------------------------------------------------------------
if __name__ == "__main__":
    path = "settings.ini"
    crudConfig(path)

```

这段代码首先检查配置文件的路径是否存在。如果没有，那么它使用我们之前创建的 **createConfig** 函数来创建它。接下来，我们创建一个 ConfigParser 对象，并向其传递要读取的配置文件路径。为了读取配置文件中的选项，我们调用我们的 ConfigParser 对象的 **get** 方法，向它传递节名和选项名。这将返回选项的值。如果你想改变一个选项的值，那么你可以使用 **set** 方法，在这里你传递节名、选项名和新值。最后，您可以使用 **remove_option** 方法来删除一个选项。

在我们的示例代码中，我们将 font_size 的值更改为“12 ”,并完全删除了 font_style 选项。然后，我们将更改写回磁盘。

虽然这不是一个很好的例子，但是你不应该有一个函数像这个函数一样做所有的事情。所以让我们把它分成一系列的功能:

```py

import ConfigParser
import os

def create_config(path):
    """
    Create a config file
    """
    config = ConfigParser.ConfigParser()
    config.add_section("Settings")
    config.set("Settings", "font", "Courier")
    config.set("Settings", "font_size", "10")
    config.set("Settings", "font_style", "Normal")
    config.set("Settings", "font_info",
               "You are using %(font)s at %(font_size)s pt")

    with open(path, "wb") as config_file:
        config.write(config_file)

def get_config(path):
    """
    Returns the config object
    """
    if not os.path.exists(path):
        create_config(path)

    config = ConfigParser.ConfigParser()
    config.read(path)
    return config

def get_setting(path, section, setting):
    """
    Print out a setting
    """
    config = get_config(path)
    value = config.get(section, setting)
    print "{section} {setting} is {value}".format(
        section=section, setting=setting, value=value)
    return value

def update_setting(path, section, setting, value):
    """
    Update a setting
    """
    config = get_config(path)
    config.set(section, setting, value)
    with open(path, "wb") as config_file:
        config.write(config_file)

def delete_setting(path, section, setting):
    """
    Delete a setting
    """
    config = get_config(path)
    config.remove_option(section, setting)
    with open(path, "wb") as config_file:
        config.write(config_file)

#----------------------------------------------------------------------
if __name__ == "__main__":
    path = "settings.ini"
    font = get_setting(path, 'Settings', 'font')
    font_size = get_setting(path, 'Settings', 'font_size')

    update_setting(path, "Settings", "font_size", "12")

    delete_setting(path, "Settings", "font_style")

```

与第一个例子相比，这个例子进行了大量的重构。我甚至用 PEP8 来命名函数。每个函数都应该是自解释和自包含的。我们没有将所有的逻辑放入一个函数中，而是将它分成多个函数，然后在底层的 if 语句中演示它们的功能。现在您可以导入该模块并自己使用它。

请注意，本例中的部分是硬编码的，因此您需要进一步更新本例，使其完全通用。

### 如何使用插值

ConfigParser 模块还允许插值，这意味着您实际上可以使用一些选项来构建另一个选项。我们实际上是用 **font_info** 选项来实现的，因为它的值是基于 font 和 font_size 选项的。我们实际上可以使用 Python 字典来更改插值。让我们花点时间来证明这两个事实。

```py

import ConfigParser

#----------------------------------------------------------------------
def interpolationDemo(path):
    """"""
    if not os.path.exists(path):
        createConfig(path)

    config = ConfigParser.ConfigParser()
    config.read(path)

    print config.get("Settings", "font_info")

    print config.get("Settings", "font_info", 0,
                     {"font": "Arial", "font_size": "100"})

#----------------------------------------------------------------------
if __name__ == "__main__":
    path = "settings.ini"
    interpolationDemo(path)

```

如果运行此代码，您应该会看到类似如下的输出:

```py

You are using Courier at 12 pt
You are using Arial at 100 pt

```

### 包扎

至此，您应该对 ConfigParser 的功能有了足够的了解，可以在自己的项目中使用它。还有另一个名为 [ConfigObj](http://www.voidspace.org.uk/python/configobj.html) 的项目，它不是 Python 的一部分，您可能也想看看。ConfigObj 比 ConfigParser 更灵活，功能更多。但是如果您有困难或者您的组织不允许第三方软件包，那么 ConfigParser 可能会满足您的要求。

### 附加阅读

*   官方 ConfigParser [文档](http://docs.python.org/2/library/configparser.html)
*   简要的 ConfigObj [教程](https://www.blog.pythonlibrary.org/2010/01/01/a-brief-configobj-tutorial/)
*   [ConfigObj + wxPython =极客快乐](https://www.blog.pythonlibrary.org/2010/01/17/configobj-wxpython-geek-happiness/)