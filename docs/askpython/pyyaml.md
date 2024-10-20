# 使用 PyYAML 进行 Python YAML 处理

> 原文：<https://www.askpython.com/python-modules/pyyaml>

YAML 代表 YAML 标记语言。它广泛用于为许多不同的 DevOps 工具和应用程序编写和存储配置文件。它以文本文件的形式编写，易于人类阅读，易于阅读和理解。它使用**。yaml** 或**。yml** 作为扩展。它类似于其他数据序列化语言，如 JSON 和 XML。

***数据序列化**是通过网络执行配置文件传输和恢复的标准格式。*

使用 YAML 用 Python 编写的数据序列化文件可以很容易地通过网络发送，然后可以使用另一种编程语言对其进行反序列化。它支持多种语言，如 Python、JavaScript、Java 等等。这些语言有 YAML 库，使它们能够解析和使用 YAML 文件。在本文中，我们将使用 Python 通过一些例子来回顾 YAML。它还支持列表、字典和数组等数据结构。

## YAML vs XML vs JSON

让我们看一个配置文件的例子，看看三个版本，以获得概述和语法。

##### YAML——YAML 不是一种标记语言

```py
configuration:
  name: my-api-config
  version: 1.0.0
  about: "some description"
  description: |
    Lorem ipsum dolor sit amet, 
    consectetur adipiscing elit
  scripts:
    "start:dev": main.py
  keywords: ["hello", "there"]

```

##### XML–可扩展标记语言

```py
<configuration>
  <name>my-api-config</name>
  <version>1.0.0</version>
  <about>some description</about>
  <description>Lorem ipsum dolor sit amet, consectetur adipiscing elit</description>
  <scripts>
    <start:dev>main.py</start:dev>
  </scripts>
  <keywords>hello</keywords>
  <keywords>there</keywords>
</configuration>

```

##### JSON–JavaScript 对象符号

```py
{
    "configuration": {
        "name": "my-api-config",
        "version": "1.0.0",
        "about": "some description",
        "description": "Lorem ipsum dolor sit amet, \nconsectetur adipiscing elit\n",
        "scripts": {
            "start:dev": "main.py"
        },
        "keywords": [
            "hello",
            "there"
        ]
    }
}

```

## 破解一份 YAML 的文件

我们示例中的 YAML 文件显示了应用程序的一些配置设置。与其他两种配置文件格式相比，有一些明显的区别，它们甚至使用了更多的符号解析。YAML 的核心是使用**键:值**对在文件中存储数据。这些键应该是字符串，它们可以用引号或者不用引号来写。这些值可以接受多种数据类型，如整数、字符串、列表、布尔值等。

写入 YAML 文件时，需要正确的**缩进**。使用制表符是不允许的，所以我们需要小心，否则我们将有林挺错误在我们的文件中。所以，它真的很简单，也很易读。不像 XML 或 JSON 文件，我们不必在阅读时解析多个符号。

```py
configuration:
  name: my-api-config
  version: 1.0.0
  about: "some description"
  # This is a comment
  description: |
    Lorem ipsum dolor sit amet, 
    consectetur adipiscing elit
  scripts:
    "start:dev": main.py
  keywords: ["hello", "there"]

```

我们还可以像上面写的那样在文件中包含**注释**，以及使用 **`|`管道字符**的多行字符串，如示例代码所示。

## 用 PyYaml 处理 YAML 文件

在本节中，我们将使用 Python 的 PyYaml 模块对 YAML 文件执行一些基本操作，如读取、写入和修改数据。

*   **安装`PyYaml`**

```py
pip install pyyaml

```

### 读取一个`yaml` 文件

假设我们有一个带有一些配置的 yaml 文件，我们想用 Python 读取其中的内容。

**文件名** : `config_one.yml`

```py
configuration:
  name: my-api-config
  version: 1.0.0
  about: some description
  stack:
    - python
    - django

```

接下来，我们将创建一个新的 python 文件，并尝试读取 yml 文件。

**文件名** : `PyYaml.py`

```py
import yaml
with open("config_one.yml", "r") as first_file:
    data = yaml.safe_load(first_file)
    print(type(data))
    print(data)
"""
Output:

<class 'dict'>
{'configuration': {'name': 'my-api-config', 'version': '1.0.0', 'about': 'some description', 'stack': ['python', 'django']}}

"""

```

**说明:**

我们正在使用 **`import yaml`** 导入`**pyyaml**` 模块。要读取一个 yaml 文件，我们首先必须以读取模式打开文件，然后使用 **`safe_load()`** 加载内容。由于不同的构造函数，比如`load()`函数，所以有多个加载器。*使用`load()` 并不安全，因为它允许执行几乎任何脚本，包括恶意代码，这一点也不安全。*因此，`safe_load()`是推荐的方式，它不会创建任何任意对象。

我们使用 Python 代码打印出 yaml 文件中的数据类型。控制台显示输出为 `**<class dict>**`，包含的数据格式化为**字典**，存储为 **`key: value`** 对。

### 修改我们的`yaml` 文件

要修改我们已经加载的文件，我们必须首先确定数据类型。如果键的值是一个字符串，我们必须在更新 `**key: value**`对之前将所有的附加值放在一个列表中。

```py
import yaml

with open("config_one.yml", "r") as first_file:
    data = yaml.safe_load(first_file)
    print(type(data))
    # Accessing our <class dict> and modifying value data using a key
    data["configuration"]["stack"] = ["flask", "sql"]
    # Appending data to the list
    data["configuration"]["stack"].append("pillow")
    print(data)

"""
Output:

<class 'dict'>
{'configuration': {'name': 'my-api-config', 'version': '1.0.0', 'about': 'some description', 'stack': ['flask', 'sql', 'pillow']}}

"""

```

**说明:**

我们在这里有一个*嵌套字典*，我们正在使用我们试图修改其值的键来访问数据。还有一个 **`append()`** 功能，用于向值列表添加另一个项目。*请注意，这些修改仅在运行时执行。我们将把这些值写入新的 yaml 文件。*

### 用修改后的数据写入一个`yaml`文件

只需几行代码，就可以将上述数据以及修改后的值写入一个新文件中。

```py
import yaml

with open("config_one.yml", "r") as first_file:
    data = yaml.safe_load(first_file)
    print(type(data))

    # Accessing our <class dict> and modifying value data using a key
    data["configuration"]["stack"] = ["flask", "sql"]

    # Appending data to the list
    data["configuration"]["stack"].append("pillow")
    print(data)

# Writing a new yaml file with the modifications
with open("new_config.yaml", "w") as new_file:
    yaml.dump(data, new_file)

```

**文件名** : `new_config.yaml`

```py
configuration:
  about: some description
  name: my-api-config
  stack:
  - flask
  - sql
  - pillow
  version: 1.0.0

```

**解释**:

我们必须使用下面的语法提供新的文件名，然后使用带有两个参数的 **`yaml.dump`** ，数据变量包含原始的 yaml 代码以及对它所做的更改，第二个参数作为声明用于执行 write 方法的`**new_file**`变量。我们可以看到，新文件保留了原始文件中的代码以及我们对其进行的更改。

## 摘要

在本文中，我们介绍了 yaml 文件的基本结构，并使用它来读取、修改和写入新文件的配置。我们还对同一个 YAML 文件使用了不同的语法，并将它与 JSON 和 XML 进行了比较。用于编写 YAML 文件的极简方法显然非常简单，易于阅读，这使得它成为各种技术栈使用的最流行的文本格式配置文件之一。

## 参考

[PyYAML 文档](https://pyyaml.org/wiki/PyYAMLDocumentation)