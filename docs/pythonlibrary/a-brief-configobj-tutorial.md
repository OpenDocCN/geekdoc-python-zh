# 简要的 ConfigObj 教程

> 原文：<https://www.blog.pythonlibrary.org/2010/01/01/a-brief-configobj-tutorial/>

Python 附带了一个方便的模块，叫做 [ConfigParser](http://docs.python.org/library/configparser.html) 。这对于创建和读取配置文件(又名 INI 文件)很有好处。然而，迈克尔·福德(《T2》的作者)和[尼古拉·拉罗萨](http://www.teknico.net/)决定编写他们自己的配置模块，名为 [ConfigObj](http://www.voidspace.org.uk/python/configobj.html) 。在许多方面，它是对标准库模块的改进。当我第一次看 ConfigObj 的主页时，我认为它有很好的文档，但它似乎没有任何完整的功能片段。由于我从 docs plus examples 中学习，我发现在没有示例的情况下开始使用 ConfigObj 会更加困难。当我开始写这篇文章时，我并不知道迈克尔·福德已经就这个主题写了自己的[教程](http://www.voidspace.org.uk/python/articles/configobj.shtml)；但是我承诺过我会写我自己的，所以这就是你今天要读的！

## 入门指南

首先，您需要[下载 ConfigObj](http://www.voidspace.org.uk/python/modules.shtml#configobj) 。一旦你下载并安装，我们可以继续。明白了吗？那我们就看看它能做什么吧！

首先，打开一个文本编辑器，创建一个包含如下内容的文件:

 `product = Sony PS3
accessories = controller, eye, memory stick
# This is a comment that will be ignored
retail_price = $400` 

把它保存在你喜欢的任何地方。我将把我的命名为“config.ini”。现在让我们看看如何使用 ConfigObj 来提取这些信息:

 `>>> from configobj import ConfigObj
>>> config = ConfigObj(r"path to config.ini")
>>> config["product"]
'Sony PS3'
>>> config["accessories"]
['controller', 'eye', 'memory stick']
>>> type(config["accessories"])` 

可以看到，ConfigObj 使用 Python 的 dict API 来访问它提取的信息。要让 ConfigObj 解析文件，只需将文件的路径传递给 ConfigObj。现在，如果信息在一个部分下(即[Sony])，那么您必须使用["Sony"]，像这样:config["Sony"]["product"]，预先挂起所有内容。还要注意“附件”部分是作为字符串列表返回的。ConfigObj 将接受任何带有逗号分隔列表的有效行，并将其作为 Python 列表返回。您也可以在配置文件中创建多行字符串，只要用三个单引号或双引号将它们括起来。

如果您需要在文件中创建一个子节，那么使用额外的方括号。比如，[索尼]是顶节，[[Playstation]]是子节，[[[PS3]]]是子节的子节。您可以创建任意深度的子部分。有关文件格式的更多信息，我推荐上面链接的文档。

现在我们将反向操作，以编程方式创建配置文件。

```py

import configobj

def createConfig(path):
    config = configobj.ConfigObj()
    config.filename = path
    config["Sony"] = {}
    config["Sony"]["product"] = "Sony PS3"
    config["Sony"]["accessories"] = ['controller', 'eye', 'memory stick']
    config["Sony"]["retail price"] = "$400"
    config.write()

```

如您所见，只需要 8 行代码。在上面的代码中，我们创建了一个函数，并将配置文件的路径传递给它。然后我们创建一个 ConfigObj 对象并设置它的*文件名*属性。为了创建这个部分，我们创建一个名为“Sony”的空 dict。然后，我们以同样的方式预处理每一行的内容。最后，我们调用配置对象的 *write* 方法将数据写入文件。

## 使用配置规范

ConfigObj 还提供了一种使用 *configspec* 来验证配置文件的方法。当我提到我将要写这篇文章时，Steven Sproat([Whyteboard](https://launchpad.net/whyteboard)的创建者)主动提供了他的 [configspec 代码](http://bazaar.launchpad.net/~sproaty/whyteboard/development/annotate/head%3A/utility.py)作为例子。我采用了他的规范，并用它来创建一个默认的配置文件。在这个例子中，我们使用 Foord 的 [validate](http://www.voidspace.org.uk/python/validate.html) 模块来进行验证。我不认为它包含在您的 ConfigObj 下载中，所以您可能也需要下载它。现在，让我们看一下代码:

```py

import configobj, validate

cfg = """
bmp_select_transparent = boolean(default=False)
canvas_border = integer(min=10, max=35, default=15)
colour1 = list(min=3, max=3, default=list('280', '0', '0'))
colour2 = list(min=3, max=3, default=list('255', '255', '0'))
colour3 = list(min=3, max=3, default=list('0', '255', '0'))
colour4 = list(min=3, max=3, default=list('255', '0', '0'))
colour5 = list(min=3, max=3, default=list('0', '0', '255'))
colour6 = list(min=3, max=3, default=list('160', '32', '240'))
colour7 = list(min=3, max=3, default=list('0', '255', '255'))
colour8 = list(min=3, max=3, default=list('255', '165', '0'))
colour9 = list(min=3, max=3, default=list('211', '211', '211'))
convert_quality = option('highest', 'high', 'normal', default='normal')
default_font = string
default_width = integer(min=1, max=12000, default=640)
default_height = integer(min=1, max=12000, default=480)
imagemagick_path = string
handle_size = integer(min=3, max=15, default=6)
language = option('English', 'English (United Kingdom)', 'Russian', 'Hindi', default='English')
print_title = boolean(default=True)
statusbar = boolean(default=True)
toolbar = boolean(default=True)
toolbox = option('icon', 'text', default='icon')
undo_sheets = integer(min=5, max=50, default=10)
"""

def createConfig(path):
    """
    Create a config file using a configspec
    and validate it against a Validator object
    """
    spec = cfg.split("\n")
    config = configobj.ConfigObj(path, configspec=spec)
    validator = validate.Validator()
    config.validate(validator, copy=True)
    config.filename = path
    config.write()

if __name__ == "__main__":
    createConfig("config.ini")

```

如果你去看看 Steven 的原始 configspec，你会注意到我把他的语言列表缩短了不少。我这样做是为了让代码更容易阅读。无论如何， [configspec](http://www.voidspace.org.uk/python/configobj.html#configspec) 允许程序员指定为配置文件中的每一行返回什么类型。它还可以用来设置默认值、最小值和最大值(等等)。如果您运行上面的代码，您会看到在当前工作目录中生成了一个“config.ini”文件，其中只有默认值。如果程序员没有指定默认值，那么这一行甚至不会被添加到配置中。

让我们仔细看看发生了什么，以确保您理解。在 *createConfig* 函数中，我们通过传入文件路径并设置 configspec 来创建一个 *ConfigObj* 实例。请注意，configspec 也可以是普通的文本文件或 python 文件，而不是本例中的字符串。接下来，我们创建一个*验证器*对象。正常的用法是只调用*config . validate(validator)*，但是在这段代码中，我将 *copy* 参数设置为 *True* ，这样我就可以创建一个文件。否则，它只会验证我传入的文件是否符合 configspec 的规则。最后，我设置配置的文件名*并写出数据。*

## 包扎

现在，您已经了解了 ConfigObj 的来龙去脉。我希望你会和我一样觉得它很有帮助。还有很多东西要学，所以一定要看看下面的一些链接。

*注意:所有代码都是在 Windows XP 上用 Python 2.5、ConfigObj 4.6.0 和 Validate 1.0.0 测试的。*

**延伸阅读**

*   [ConfigObj 官方文档](http://www.voidspace.org.uk/python/configobj.html)
*   [验证官方文件](http://www.voidspace.org.uk/python/validate.html)
*   [迈克尔·福德的 ConfigObj 教程](http://www.voidspace.org.uk/python/articles/configobj.shtml)

**下载源码**

*   [configObjTut.zip](https://www.blog.pythonlibrary.org/wp-content/uploads/2010/01/configObjTut.zip)
*   [configObjTut.tar](https://www.blog.pythonlibrary.org/wp-content/uploads/2010/01/configObjTut.tar)