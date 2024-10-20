# Python 中的 xmltodict 模块:实用参考

> 原文：<https://www.askpython.com/python-modules/xmltodict-module>

在本教程中，我们将了解如何安装 xmltodict 模块，并在我们的 Python 程序中使用它来轻松处理 XML 文件。我们将看到如何将 XML 转换成 Python 字典和 JSON 格式，反之亦然。

## 使用 pip 安装 xmltodict 模块

对于 Python 3 或更高版本，我们可以使用终端使用 [pip3 命令](https://www.askpython.com/python-modules/python-pip)来安装 xmltodict。

```py
pip3  install xmltodict

```

对于 Python 的旧版本，我们可以使用下面的命令来安装 xmltodict。

```py
pip install xmltodict

```

## 什么是 XML 文件？

XML 代表可扩展标记语言，它主要是为存储和传输数据而设计的。

它是一种支持编写结构化数据的描述性语言，我们必须使用其他软件来存储、发送、接收或显示 XML 数据。

以下 XML 文件包含飞机的数据，如年份、品牌、型号和颜色。

```py
<?xml version = "1.0" encoding = "utf-8"?>
<!-- xslplane.1.xml -->
<?xml-stylesheet type = "text/xsl"  href = "xslplane.1.xsl" ?>
<plane>
   <year> 1977 </year>
   <make> Cessna </make>
   <model> Skyhawk </model>
   <color> Light blue and white </color>
</plane>

```

现在，在下面的小节中，我们将使用这个飞机数据，看看如何将它转换成 Python dictionary 和 JSON，并使用 xmltodict 模块将它们转换回 XML 格式。

## 如何将 XML 数据读入 Python 字典？

我们可以使用 xmltodict 模块中的`xmltodict.parse()`方法将 XML 文件转换成 Python 字典。

方法将一个 XML 文件作为输入，并将其转换为有序字典。

然后，我们可以使用 Python 字典的 dict 构造函数从有序字典中提取字典数据。

```py
#import module
import xmltodict

#open the file
fileptr = open("/home/aditya1117/askpython/plane.xml","r")

#read xml content from the file
xml_content= fileptr.read()
print("XML content is:")
print(xml_content)

#change xml format to ordered dict
my_ordered_dict=xmltodict.parse(xml_content)
print("Ordered Dictionary is:")
print(my_ordered_dict)
print("Year of plane is:")
print(my_ordered_dict['plane']['year'])

#Use contents of ordered dict to make python dictionary
my_plane= dict(my_ordered_dict['plane'])
print("Created dictionary data is:")
print(my_plane)
print("Year of plane is")
print(my_plane['year'])

```

输出:

```py
XML content is:
<?xml version = "1.0" encoding = "utf-8"?>
<!-- xslplane.1.xml -->
<?xml-stylesheet type = "text/xsl"  href = "xslplane.1.xsl" ?>
<plane>
   <year> 1977 </year>
   <make> Cessna </make>
   <model> Skyhawk </model>
   <color> Light blue and white </color>
</plane>

Ordered Dictionary is:
OrderedDict([('plane', OrderedDict([('year', '1977'), ('make', 'Cessna'), ('model', 'Skyhawk'), ('color', 'Light blue and white')]))])
Year of plane is:
1977
Created dictionary data is:
{'year': '1977', 'make': 'Cessna', 'model': 'Skyhawk', 'color': 'Light blue and white'}
Year of plane is
1977

```

在上面的例子中，我们使用 **`xmltodict.parse()`** 方法成功地从 XML 格式中提取了我们的飞机数据，并以有序字典和字典的形式打印了数据。

## 如何将 Python 字典转换成 XML？

我们可以使用 xmltodict 模块的 **`xmltodict.unparse()`** 方法将 python 字典转换成 XML 格式。

这个方法接受 dictionary 对象作为输入，返回 XML 格式的数据作为输出。

这里唯一的限制是**字典应该只有一个根**，这样 XML 数据就可以很容易地被格式化。否则就会造成 **`ValueError`** 。

```py
#import module
import xmltodict

#define dictionary with all the attributes
mydict={'plane':{'year': '1977', 'make': 'Cessna', 'model': 'Skyhawk', 'color':'Light blue and white'}}
print("Original Dictionary of plane data is:")
print(mydict)

#create xml format
xml_format= xmltodict.unparse(my_ordered_dict,pretty=True)
print("XML format data is:")
print(xml_format)

```

输出:

```py
Original Dictionary of plane data is:
{'plane': {'year': '1977', 'make': 'Cessna', 'model': 'Skyhawk', 'color': 'Light blue and white'}}
XML format data is:
<?xml version="1.0" encoding="utf-8"?>
<plane>
        <year>1977</year>
        <make>Cessna</make>
        <model>Skyhawk</model>
        <color>Light blue and white</color>
</plane>

```

在上面的例子中，我们从简单的 python 字典数据中创建了 XML 格式的飞机数据。现在我们将了解如何将 XML 数据转换成 JSON 格式。

## 如何把 XML 转换成 JSON？

我们可以使用 python 中的 **xmltodict** 模块和 [**json** 模块将 XML 数据转换成 JSON 格式。在这个过程中，我们首先使用 **`xmltodict.parse()`** 方法从 XML 格式创建一个有序字典。](https://www.askpython.com/python-modules/python-json-module)

然后我们使用 **`json.dumps()`** 方法将有序字典转换为 JSON 格式，该方法将有序字典作为一个参数，并将其转换为 JSON 字符串。

```py
#import module
import xmltodict
import json

#open the file
fileptr = open("/home/aditya1117/askpython/plane.xml","r")

#read xml content from the file
xml_content= fileptr.read()
print("XML content is:")
print(xml_content)

#change xml format to ordered dict
my_ordered_dict=xmltodict.parse(xml_content)
print("Ordered Dictionary is:")
print(my_ordered_dict)
json_data= json.dumps(my_ordered_dict)
print("JSON data is:")
print(json_data)
x= open("plane.json","w")
x.write(json_data)
x.close()

```

输出:

```py
XML content is:
<?xml version = "1.0" encoding = "utf-8"?>
<!-- xslplane.1.xml -->
<?xml-stylesheet type = "text/xsl"  href = "xslplane.1.xsl" ?>
<plane>
   <year> 1977 </year>
   <make> Cessna </make>
   <model> Skyhawk </model>
   <color> Light blue and white </color>
</plane>

Ordered Dictionary is:
OrderedDict([('plane', OrderedDict([('year', '1977'), ('make', 'Cessna'), ('model', 'Skyhawk'), ('color', 'Light blue and white')]))])
JSON data is:
{"plane": {"year": "1977", "make": "Cessna", "model": "Skyhawk", "color": "Light blue and white"}}

```

在上面的例子中，我们将 XML 数据读入 **`xml_content`** ，然后 **`xmltodict.parse()`** 创建一个有序字典 **`my_ordered_dict`** ，然后使用 **`json.dumps()`** 方法从有序字典中创建 JSON 数据。

## 如何将 JSON 数据转换成 XML？

现在，让我们使用 xmltodict 模块将 JSON 数据转换为 XML 格式，首先使用 **`json.load()`** 方法将 JSON 数据转换为 Python 字典，然后使用 **`xmltodict.unparse()`** 将字典转换为 XML。

同样，这里的限制是 **JSON 数据应该有一个单独的根**，否则会导致 **`ValueError`** 。

```py
#import module
import xmltodict
import json

#define dictionary with all the attributes
fileptr = open("/home/aditya1117/askpython/plane.json","r")
json_data=json.load(fileptr)
print("JSON data is:")
print(json_data)

#create xml format
xml_format= xmltodict.unparse(json_data,pretty=True)
print("XML format data is:")
print(xml_format)

```

输出:

```py
JSON data is:
{'plane': {'year': '1977', 'make': 'Cessna', 'model': 'Skyhawk', 'color': 'Light blue and white'}}
XML format data is:
<?xml version="1.0" encoding="utf-8"?>
<plane>
        <year>1977</year>
        <make>Cessna</make>
        <model>Skyhawk</model>
        <color>Light blue and white</color>
</plane>

```

在上面的例子中， **`json.load()`** 接受文件对象作为参数并解析数据，从而创建一个 Python 字典，该字典存储在 **`json_data`** 中。然后我们使用 **`xmltodict.unparse()`** 方法将字典转换成 XML 文件。

## 结论

在本文中，我们使用了 xmltodict 模块来处理 XML 数据。我们已经看到了如何将 XML 数据转换成 Python 字典和 JSON 格式，还将它们转换回 XML 格式。快乐学习！