# Python 语言规则

> 原文：<https://www.pythonforbeginners.com/cheatsheet/python-style-guide-part-1>

PEP8 已经成为大多数项目的风格指南。它提倡一种可读性很强、赏心悦目的编码风格。这是 Python 社区中一个公认的约定，通常我们应该遵循这些约定。

风格指南是关于一致性的。

与本风格指南保持一致非常重要。

项目内部的一致性更重要。

一个模块或功能的一致性是最重要的。

谷歌整理了一份非常不错的风格指南摘要，可以在这里找到:[http://Google-style guide . Google code . com/SVN/trunk/py guide . html](https://google.github.io/styleguide/pyguide.html "Google Python Style Guide")

每个样式点都有一个摘要。

### Python 语言规则

#### pychecker

*   对您的代码运行 pychecker。

#### 进口

*   仅对包和模块使用导入。

#### 包装

*   使用模块的完整路径名位置导入每个模块。

#### 例外

*   例外是允许的，但必须小心使用。

#### 全局变量

*   避免全局变量。

#### 嵌套/局部/内部类和函数

*   嵌套/局部/内部类和函数都可以。

#### 列出理解

*   对于简单的情况可以使用。

#### 默认迭代器和运算符

*   对支持迭代器和操作符的类型使用默认的迭代器和操作符，比如列表、字典和文件。

#### 发电机

*   根据需要使用发电机。

#### λ函数

*   好吧，一句话。

#### 条件表达式

*   好吧，一句话。

#### 默认参数值

*   大多数情况下没问题。

#### 性能

*   在通常使用简单、轻量级的访问器或设置器方法的地方，使用属性来访问或设置数据。

#### 真/假评估

*   尽可能使用“隐式”false。

#### 不推荐使用的语言功能

*   尽可能使用字符串方法而不是字符串模块。使用函数调用语法而不是 apply。使用 [list comprehensions](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python) 和 for 循环，而不是 filter 和 map，因为无论如何函数参数都是内联的 lambda。使用 for 循环代替 reduce。

#### 词法范围

*   可以使用。

#### 函数和方法装饰器

*   当有明显优势时，明智地使用装饰者。

#### 穿线

*   不要依赖内置类型的原子性。

#### 电源功能

*   避免这些功能。