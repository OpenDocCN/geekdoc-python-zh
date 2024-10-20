# Python:当今糟糕的代码(2013 年 10 月 30 日版)

> 原文：<https://www.blog.pythonlibrary.org/2013/10/30/python-bad-code-of-the-day-oct-30th-2013-ed/>

我们都会时不时地写出糟糕的代码。当我在解释器或调试器中试验时，我倾向于故意这样做。我说不出我为什么要这样做，除了看看我能在一行代码里放些什么很有趣。总之，本周我需要解析一个文件，如下所示:

```py

00000       TM        YT                                                TSA1112223  0000000000000000000000000020131007154712PXXXX_Name_test_test_section.txt
data line 1
data line 2

```

 你可以在这里[下载](https://www.blog.pythonlibrary.org/wp-content/uploads/2013/10/test.txt)如果你不想滚动上面的片段。无论如何，我们的目标是解析出文件头中的文件名部分，即上面的“section”。文件名本身就是**XXXX _ 名字 _ 测试 _ 测试 _ 章节. txt** 。文件名的这一部分用于标识数据库中的某些内容。所以在 Python 解释器中，我想出了下面这个方法:

```py

header.split("|")[0][125:].split("_")[-1].split(".")[0]

```

让我们给它更多的背景。上面的代码片段适合这样一个函数:

```py

#----------------------------------------------------------------------
def getSection(path):
    """"""
    with open(path) as inPath:
        lines = inPath.readlines()

    header = lines.pop(0)
    section = header.split("|")[0][125:].split("_")[-1].split(".")[0]

```

那是相当难看的代码，需要一个相当长的注释来解释我在做什么。并不是说我一开始就打算用它，但是写起来很有趣。总之，我最后把它分解成一些代码，大概如下:

```py

#----------------------------------------------------------------------
def getSection(path):
    """"""
    with open(path) as inPath:
        lines = inPath.readlines()

    header = lines.pop(0)
    filename = header[125:256].strip()
    fname = os.path.splitext(filename)[0]
    section = fname.split("_")[-1]

    print section
    print header.split("|")[0][125:].split("_")[-1].split(".")[0]

#----------------------------------------------------------------------
if __name__ == "__main__":
    path = "test.txt"
    getSection(path)

```

我在上面的例子中留下了“坏的”代码，以表明结果是相同的。是的，它仍然有点难看，但是它更容易理解，代码是自文档化的。

你最近写了什么时髦的代码？