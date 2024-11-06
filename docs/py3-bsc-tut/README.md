

# Python 3 基础教程

(c) 2015 Dr. Kristian Rother (krother@academis.eu)

由 Allegra Via、Kaja Milanowska 和 Anna Philips 贡献

根据知识共享署名-相同方式共享 4.0 许可条件分发

本文档的来源可以在[`github.com/krother/Python3_Basics_Tutorial`](https://github.com/krother/Python3_Basics_Tutorial)找到

# 介绍

## 这个教程适合谁？

这是一个面向初学者的教程。写这个教程时，我想到的是你，作为学习者，如果：

+   你曾经稍微接触过其他编程语言，比如 R、MATLAB 或 C。

+   你根本没有编程经验

+   你对 Python 很熟悉，并且想教别人

如果你按照章节和练习逐步进行，这个教程效果最好。

## 美国婴儿姓名数据集

![Babynamen](img/baby.png)

美国当局自 1880 年以来记录了所有出生为美国公民的人的名字。该数据集可以在[`www.ssa.gov/oact/babynames/limits.html`](http://www.ssa.gov/oact/babynames/limits.html)上公开获取。然而，为了保护隐私，只有至少出现 5 次的名字才会出现在数据中。

在整个教程中，我们将使用这些数据。

## 如果你是一名有经验的程序员

如果你精通任何编程语言，这个教程对你来说可能非常简单。当然，你可以通过练习来熟悉 Python 的语法。然而，这个教程几乎没有涉及 Python 的更高抽象级别，比如类、命名空间甚至函数。

对于非初学者的教程，我推荐以下免费在线书籍：

+   [学习 Python 的艰难之路](http://learnpythonthehardway.org/) - 由**泽德·肖**撰写的一种类似训练营风格的教程

+   [如何像计算机科学家一样思考](http://www.greenteapress.com/thinkpython/) - 由**艾伦·B·唐尼**撰写的非常系统化、科学化的教程

+   [深入 Python 3](http://www.diveintopython3.net/) - 每章解释一个复杂程序 - **作者马克·皮尔格里姆**
