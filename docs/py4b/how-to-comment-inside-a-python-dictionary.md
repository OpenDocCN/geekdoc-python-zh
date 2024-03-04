# 如何在 Python 字典中进行注释

> 原文：<https://www.pythonforbeginners.com/comments/how-to-comment-inside-a-python-dictionary>

python 中的注释在增加代码的可读性和可维护性方面非常方便。一般来说，我们使用注释来描述函数和类描述，用于文档目的或解释为什么在源代码中编写语句，但可能有这样的情况，我们需要解释为什么我们在字典或列表中包含某些数据。在本文中，我们将看到 python 字典的基本功能，并尝试理解如何在 python 字典中添加注释。

## python 字典的工作

在 Python 中，字典是用于以键和值对的形式存储数据的数据结构。python 字典是使用花括号定义的，键和值对在初始化时插入字典中，用冒号`":"`分隔，或者可以在初始化后使用赋值语句添加。

python 字典中的键值对被称为项目。

初始化字典的最简单方法如下:

```py
website_details={"name":"Pyhton For Beginners",
         "domain":"pythonforbeginners.com"
        }
print("dictionary is:")
print(website_details)
print("Keys in the dictionary are:")
print(website_details.keys())
print("values in the dictionary are:")
print(website_details.values())
print("itmes in the dictionay are:")
print(website_details.items())
```

输出:

```py
dictionary is:
{'name': 'Pyhton For Beginners', 'domain': 'pythonforbeginners.com'}
Keys in the dictionary are:
dict_keys(['name', 'domain'])
values in the dictionary are:
dict_values(['Pyhton For Beginners', 'pythonforbeginners.com'])
itmes in the dictionay are:
dict_items([('name', 'Pyhton For Beginners'), ('domain', 'pythonforbeginners.com')])
```

在上面的例子中，我们可以看到使用花括号定义了一个 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)，在字典中指定了键-值对，它们由冒号“:”分隔。

使用`dict_name.keys()`可以获得字典的键。类似地，字典中的值可以使用`dict_name.values()`获得，所有的项(键-值对)可以使用`dict_name.items()`方法获得。

如果我们想在字典中插入新的键值对，我们可以这样做。

```py
website_details={"name":"Pyhton For Beginners",
         "domain":"pythonforbeginners.com"
        }
#add new item to list
website_details["acronym"]="PFB"
print("dictionary is:")
print(website_details)
print("Keys in the dictionary are:")
print(website_details.keys())
print("values in the dictionary are:")
print(website_details.values())
print("itmes in the dictionay are:")
print(website_details.items())
```

输出:

```py
 dictionary is:
{'name': 'Pyhton For Beginners', 'domain': 'pythonforbeginners.com', 'acronym': 'PFB'}
Keys in the dictionary are:
dict_keys(['name', 'domain', 'acronym'])
values in the dictionary are:
dict_values(['Pyhton For Beginners', 'pythonforbeginners.com', 'PFB'])
itmes in the dictionay are:
dict_items([('name', 'Pyhton For Beginners'), ('domain', 'pythonforbeginners.com'), ('acronym', 'PFB')])
```

在上面的例子中，我们添加了一个键为“`acronym`”、值为“`PFB`”的项目。我们可以像下面的代码片段所示的`dict_name[key_name]` 一样访问与字典的键相关的值。

```py
website_details={"name":"Pyhton For Beginners",
         "domain":"pythonforbeginners.com"
        }
#add new item to list
website_details["acronym"]="PFB"
print(website_details["domain"])
```

输出:

```py
pythonforbeginners.com
```

## 在 python 中处理单行注释

通过用符号`#`初始化注释文本，可以编写一行 [python 注释](https://www.pythonforbeginners.com/comments/comments-in-python)。单行注释在源代码中遇到换行符时终止。

我们可以通过在一个新行上开始来放置单行注释，或者我们可以通过在源代码中的一个语句后放置`#`符号来放置单行注释，但是应该记住，当在源代码中发现一个新行或换行符时，注释终止。它可以在下面的源代码中可视化。

```py
website_details={"name":"Pyhton For Beginners",
         "domain":"pythonforbeginners.com"
        }
#This is single line comment from start of line
website_details["acronym"]="PFB"#this is a single line comment after an statement
print(website_details["domain"])
```

## python 中多行注释的处理

从理论上讲，python 中不存在多行注释。但是我们可以在 python 中使用单行注释和三重引号字符串实现多行注释。

我们可以使用单行注释实现多行注释，只要遇到换行符就插入一个`#`符号。这样，多行注释就被描述为一系列单行注释。

```py
website_details={"name":"Pyhton For Beginners",
         "domain":"pythonforbeginners.com"
        }
#This is a multi line comment
#implemented using # sign
website_details["acronym"]="PFB"
print(website_details["domain"])
```

如果我们不把字符串赋给任何变量，我们也可以把它们作为多行注释。当字符串没有被赋给任何变量时，它们会被解释器解析和评估，但不会生成字节码，因为没有地址可以赋给字符串。这将影响字符串作为注释的工作。在这种方法中，可以使用三重引号声明多行注释。这可以看如下。

```py
website_details={"name":"Pyhton For Beginners",
         "domain":"pythonforbeginners.com"
        }
"""This is a multiline comment
implemented with the help of 
triple quoted strings"""
website_details["acronym"]="PFB"
print(website_details["domain"])
```

## 在 python 字典中添加单行注释

我们可以像在其他地方一样，使用`#`符号在 python 字典中添加单行注释。我们只需要将注释后的内容移到新的一行，这样字典的内容就不会被注释掉。

```py
website_details={"name":"Pyhton For Beginners",
                 #This is a single line comment inserted inside a dictionary 
         "domain":"pythonforbeginners.com"
        }

website_details["acronym"]="PFB"
print(website_details["domain"])
```

## 在 python 字典中添加多行注释

我们可以只使用`#`符号在 python 字典中添加多行注释。理论上，我们只能在 python 字典中添加单行注释，但是我们可以使用连续的单行注释来模拟多行注释。

```py
website_details={"name":"Pyhton For Beginners",
                 #This is a multiline comment inside a dictionary
                 #inserted with the help of consecutive single line comments
         "domain":"pythonforbeginners.com"
        }

website_details["acronym"]="PFB"
print(website_details["domain"])
```

在 python 字典中使用三重引号字符串作为注释是行不通的，python 解释器会抛出错误。

```py
website_details={"name":"Pyhton For Beginners",
                 """This is a multiline comment inside a dictionary
                 inserted with the help of triple quoted strings and
                 it will cause error"""
         "domain":"pythonforbeginners.com"
        }

website_details["acronym"]="PFB"
print(website_details["domain"])
```

## 结论

在本文中，我们已经看到了 python 字典的工作方式，以及 python 中的单行注释和多行注释，然后我们尝试实现了在 python 字典中插入单行注释和多行注释的方法。请继续关注更多内容丰富的文章。