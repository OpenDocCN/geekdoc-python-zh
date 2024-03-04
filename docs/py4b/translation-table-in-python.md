# Python 中的翻译表

> 原文：<https://www.pythonforbeginners.com/basics/translation-table-in-python>

Python 为我们提供了操作字符串的不同方式。在本文中，我们将讨论转换表，并使用它在 python 中用字符串中的另一个字符替换一个字符。

## 什么是翻译表？

简单地说，翻译表是一个字符到另一个字符的映射。在处理字符串时，我们可能需要用字符串中的另一个字符替换一个字符。在这种情况下，我们可以使用转换表来确定哪个字符必须被哪个字符替换。

您可以将转换表想象成一个字典，其中的键是最初出现在字符串中的字符，值是将替换现有字符的字符。

现在，让我们看看如何用 python 创建一个翻译表。

## 如何创建翻译表？

Python 为我们提供了 maketrans()函数，用它我们可以创建一个翻译表。maketrans()函数接受三个参数并返回一个转换表。第一个参数是包含需要替换的字符的字符串。第二个输入参数是包含新字符的字符串。第三个也是可选的参数是一个字符串，它包含需要从任何字符串中删除的字符。maketrans()函数的语法如下:

`maketrans(old_characters,new_characters,characters_to_delete).` 在这里，

*   **old_characters** 是包含需要替换的字符的字符串。
*   **new_characters** 是一个字符串，它包含的字符将被用来代替 **old_characters** 中的字符。理想情况下，**新字符**的长度应该等于**旧字符**。这样， **old_characters** 中的每个字符都会映射到 new_character 中相应位置的字符上。
*   **characters_to_delete** 包含将从任何字符串中删除的字符。

我们可以用 python 创建一个翻译表，如下例所示。

```py
import string
input_string = """This is PythonForBeginners.com.
Here, you   can read python tutorials for free."""
translation_table = input_string.maketrans("abcdefghijklmnopqrstupwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
print("The translation table is:")
print(translation_table) 
```

输出:

```py
The translation table is:
{97: 65, 98: 66, 99: 67, 100: 68, 101: 69, 102: 70, 103: 71, 104: 72, 105: 73, 106: 74, 107: 75, 108: 76, 109: 77, 110: 78, 111: 79, 112: 86, 113: 81, 114: 82, 115: 83, 116: 84, 117: 85, 119: 87, 120: 88, 121: 89, 122: 90} 
```

这里，字符 A 到 Z 和 A 到 Z 的 ASCII 值用于创建翻译表。转换表的关键字是小写字母的 ASCII 值，对应的值是大写字母的 ASCII 值。我们可以使用这个转换表将小写字符替换为大写字符。我们没有指定任何必须删除的字符。

## 翻译表怎么用？

我们使用翻译表和 translate()方法来替换字符串中的字符。在字符串上调用 translate()方法时，它将翻译表作为输入，并根据翻译表替换原始字符串中的字符。你可以从下面的例子中理解这一点。

```py
import string
input_string = """This is PythonForBeginners.com.
Here, you   can read python tutorials for free."""
translation_table = input_string.maketrans("abcdefghijklmnopqrstupwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
new_string = input_string.translate(translation_table)
print("The original string is:")
print(input_string)
print("Output String is:")
print(new_string) 
```

输出:

```py
The original string is:
This is PythonForBeginners.com.
Here, you   can read python tutorials for free.
Output String is:
THIS IS PYTHONFORBEGINNERS.COM.
HERE, YOU   CAN READ VYTHON TUTORIALS FOR FREE.
```

这里，我们使用上一个示例中创建的转换表将所有小写字符替换为大写字符。

## 结论

在本文中，我们讨论了 python 中的翻译表。我们还看到了如何使用 maketrans()方法和 translate()方法来替换字符串中的字符。要了解更多关于 python 编程的知识，你可以阅读这篇关于[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。你可能也会喜欢这篇关于 Python 中[链表的文章。](https://www.pythonforbeginners.com/lists/linked-list-in-python)