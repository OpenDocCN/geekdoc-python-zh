# Python 3.10 -带括号的上下文管理器

> 原文：<https://www.blog.pythonlibrary.org/2021/09/08/python-3-10-parenthesized-context-managers/>

Python 3.10 将于下个月发布，所以现在是时候开始谈论它将包含的新特性了。Python 的核心开发者最近宣布，他们正在添加*带括号的上下文管理器*，这是对 Python 3.9 和更早版本不支持跨行延续括号的错误的[错误修复](https://bugs.python.org/issue12782)。

您仍然可以有多个上下文管理器，但是它们不能在括号中。基本上，它使代码看起来有点笨拙。下面是一个使用 **Python 3.8** 的例子:

```py
>>> with (open("test_file1.txt", "w") as test, 
          open("test_file2.txt", "w") as test2): 
        pass                                                                 

File "", line 1
    with (open("test_file1.txt", "w") as test,
                                      ^
SyntaxError: invalid syntax

```

有趣的是，尽管没有得到官方支持，这些代码在 Python 3.9 中似乎也能正常工作。如果你去读最初的[错误报告](https://bugs.python.org/issue12782)，那里提到添加 [PEG 解析器(PEP 617)](https://www.python.org/dev/peps/pep-0617/) 应该可以解决这个问题。

这有点让人想起 Python 字典是无序的，直到在 [Python 3.6](https://docs.python.org/3.6/whatsnew/3.6.html#new-dict-implementation) 中一个实现细节使它们有序，但直到 Python 3.8 才正式有序。

无论如何，Python 3.10 已经批准了这一更改，这使得以下所有示例代码都是有效的:

```py
with (CtxManager() as example):
    ...

with (
    CtxManager1(),
    CtxManager2()
):
    ...

with (CtxManager1() as example,
      CtxManager2()):
    ...

with (CtxManager1(),
      CtxManager2() as example):
    ...

with (
    CtxManager1() as example1,
    CtxManager2() as example2
):
    ...
```

你可以在 Python 3.10 的[新特性章节](https://docs.python.org/3.10/whatsnew/3.10.html#parenthesized-context-managers)中了解更多关于这个新“特性”的信息。