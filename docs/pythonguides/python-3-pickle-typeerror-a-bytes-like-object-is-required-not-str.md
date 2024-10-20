# Python 3 pickle typeerror 需要类似字节的对象，而不是“str”

> 原文：<https://pythonguides.com/python-3-pickle-typeerror-a-bytes-like-object-is-required-not-str/>

[![Python & Machine Learning training courses](img/49ec9c6da89a04c9f45bab643f8c765c.png)](https://sharepointsky.teachable.com/p/python-and-machine-learning-training-course)

在这个 [Python 教程](https://pythonguides.com/python-download-and-installation/)中，我们将讨论如何修复 **Python 3 pickle typeerror 需要一个类似字节的对象而不是‘str’**错误。

最近，我试图使用 python 中的 **pickle 模块读取一个文件。我得到一个错误:**类型错误:需要一个类似字节的对象，而不是‘str’**。下面是我用来读取文件的代码。**

```py
import pickle              

file = open('student.p', 'r')
student = pickle.load(file)      
file.close()                       

print(student)
```

您可以看到错误截图，pickle **typeerror 需要一个类似字节的对象，而不是出现“str”**错误。

![pickle error](img/13c0a6fd8189f789148a9e7ef0fd0683.png "pickle error")

Python 3 pickle typeerror a bytes-like object is required not str

## typeerror 需要类似字节的对象，而不是“str”python 3

解决方案很简单，这里我们必须使用**‘Rb’**而不是下面一行代码中的**‘r’**:

```py
file = open('student.p', 'rb')
```

完整的代码如下所示:

```py
import pickle              

file = open('student.p', 'rb')
student = pickle.load(file)      
file.close()                       

print(student)
```

现在，当您执行代码时，不会出现错误**type error a bytes-like object required not ' str ' python 3**。

您可能会喜欢以下 Python 教程:

*   [Python 读取 CSV 文件并写入 CSV 文件](https://pythonguides.com/python-read-csv-file/)
*   [Python 从路径中获取文件名](https://pythonguides.com/python-get-filename-from-the-path/)
*   [Python 读取 excel 文件并在 Python 中写入 Excel](https://pythonguides.com/python-read-excel-file/)
*   [Python 输入和 raw_input 函数](https://pythonguides.com/python-input-and-raw_input-function/)
*   [在 Python 中使用 JSON 数据](https://pythonguides.com/json-data-in-python/)
*   [Python – stderr, stdin and stdout](https://pythonguides.com/python-stderr-stdin-and-stdout/)
*   [Python 二分搜索法和线性搜索](https://pythonguides.com/python-binary-search/)
*   [Python 点积和叉积](https://pythonguides.com/python-dot-product/)
*   [Python 退出命令(quit()、exit()、sys.exit())](https://pythonguides.com/python-exit-command/)

此外，它将修复以下错误:

*   typeerror 需要类似字节的对象，而不是“str”python 3
*   python 3 pickle typeerror 需要类似字节的对象，而不是“str”
*   typeerror 需要类似字节的对象，而不是“str”python 3 split
*   python 3 替换类型错误需要类似字节的对象，而不是“str”
*   python 3 csv 类型错误需要类似字节的对象，而不是“str”

![Bijay Kumar MVP](img/9cb1c9117bcc4bbbaba71db8d37d76ef.png "Bijay Kumar MVP")[Bijay Kumar](https://pythonguides.com/author/fewlines4biju/)

Python 是美国最流行的语言之一。我从事 Python 工作已经有很长时间了，我在与 Tkinter、Pandas、NumPy、Turtle、Django、Matplotlib、Tensorflow、Scipy、Scikit-Learn 等各种库合作方面拥有专业知识。我有与美国、加拿大、英国、澳大利亚、新西兰等国家的各种客户合作的经验。查看我的个人资料。

[enjoysharepoint.com/](https://enjoysharepoint.com/)[](https://www.facebook.com/fewlines4biju "Facebook")[](https://www.linkedin.com/in/fewlines4biju/ "Linkedin")[](https://twitter.com/fewlines4biju "Twitter")