# 如何用 Python 将字典保存到文件中

> 原文：<https://www.pythonforbeginners.com/dictionary/how-to-save-dictionary-to-file-in-python>

python 字典用于存储程序中的键值映射。有时，我们可能需要将字典直接存储在文件中。在本文中，我们将讨论如何用 Python 将字典直接保存到文件中。

## 使用 Python 中的字符串将字典保存到文件中

要将字典保存到文件中，我们可以先将字典转换为字符串。之后，我们可以将字符串保存在文本文件中。为此，我们将遵循以下步骤。

*   首先，我们将使用 str()函数将字典转换为字符串。str()函数接受一个对象作为输入，并返回它的字符串表示。
*   获得字典的字符串表示后，我们将使用 open()函数以写模式打开一个文本文件。open()函数将文件名和模式作为输入参数，并返回一个文件流对象，比如 myFile。
*   在获得文件流对象 myFile 之后，我们将使用 write()方法将字符串写入文本文件。在 file 对象上调用 write()方法时，该方法将一个字符串作为输入参数，并将其写入文件。
*   在执行 write()方法之后，我们将使用 close()方法关闭文件流。

按照上面的步骤，您可以将字典以字符串形式保存到文件中。将字典保存到文件后，您可以通过打开文件来验证文件内容。在下面的代码中，我们首先将一个 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)保存到一个文件中。

```py
myFile = open('sample.txt', 'w')
myDict = {'Roll': 4, 'Name': 'Joel', 'Language': 'Golang'}
print("The dictionary is:")
print(myDict)
myFile.write(str(myDict))
myFile.close()
myFile = open('sample.txt', 'r')
print("The content of the file after saving the dictionary is:")
print(myFile.read())
```

输出:

```py
The dictionary is:
{'Roll': 4, 'Name': 'Joel', 'Language': 'Golang'}
The content of the file after saving the dictionary is:
{'Roll': 4, 'Name': 'Joel', 'Language': 'Golang'}
```

## 用 Python 将字典以二进制格式保存到文件中

我们可以直接以二进制格式存储字典，而不是以文本格式存储字典。为此，我们将使用 Python 中的 pickle 模块。为了使用 pickle 模块将字典保存到文件中，我们将遵循以下步骤。

*   首先，我们将使用 open()函数以写二进制(wb)模式打开一个文件。open()函数将文件名和模式作为输入参数，并返回一个文件流对象，比如 myFile。
*   pickle 模块为我们提供了 dump()方法，在该方法的帮助下，我们可以将二进制格式的字典保存到文件中。dump()方法将一个对象作为第一个输入参数，将一个文件流作为第二个输入参数。执行后，它将对象以二进制格式保存到文件中。我们将把字典作为第一个参数，把 myFile 作为第二个输入参数传递给 dump()方法。
*   在执行 dump()方法之后，我们将使用 close()方法关闭文件。

以下是用 python 将字典保存到文件中的 python 代码。

```py
import pickle

myFile = open('sample_file', 'wb')
myDict = {'Roll': 4, 'Name': 'Joel', 'Language': 'Golang'}
print("The dictionary is:")
print(myDict)
pickle.dump(myDict,myFile)
myFile.close()
```

将字典保存为二进制格式后，我们可以使用 pickle 模块中的 load()方法检索它。load()方法将包含二进制形式的 python 对象的文件流作为其输入参数，并返回 Python 对象。使用 dump()方法将字典保存到文件后，我们可以从文件中重新创建字典，如下所示。

```py
import pickle

myFile = open('sample_file', 'wb')
myDict = {'Roll': 4, 'Name': 'Joel', 'Language': 'Golang'}
print("The dictionary is:")
print(myDict)
pickle.dump(myDict,myFile)
myFile.close()
myFile = open('sample_file', 'rb')
print("The content of the file after saving the dictionary is:")
print(pickle.load(myFile))
```

输出:

```py
The dictionary is:
{'Roll': 4, 'Name': 'Joel', 'Language': 'Golang'}
The content of the file after saving the dictionary is:
{'Roll': 4, 'Name': 'Joel', 'Language': 'Golang'}
```

## 结论

在本文中，我们讨论了用 python 将字典保存到文件的两种方法。想了解更多关于字典的知识，可以阅读这篇关于 python 中[字典理解的文章。](https://www.pythonforbeginners.com/dictionary/dictionary-comprehension-in-python)