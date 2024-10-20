# Django 模板语言——初学者入门

> 原文：<https://www.askpython.com/django/django-template-language>

在本文中，我们将学习 Django 模板语言以及如何在模板中使用它。

## 什么是 Django 模板语言？

**Django 模板语言或 DTL** 是一种基于文本的模板语言，在 HTML、CSS、JS 等脚本之间提供桥梁。以及像 python 这样的编程语言。

**DTL** 专为开发者打造，用于将 Django 逻辑代码嵌入 HTML 模板文件。

与其他基于文本的模板语言相比，DTL 也有很大的优势，因为它

*   简单
*   简单易学的语法
*   展开性

## 为什么我们需要 Django 模板语言？

web 应用程序有两个主要组件:

1.  **前端**
2.  **后端**

因此，如果前端开发人员分别负责 HTML 部分，而后端开发人员分别负责 Python-Django 部分，那就更有意义了。

Django 模板语言可以做到这一点！！

有了 **DTL** ，前端开发者不需要知道 python，后端程序员也不需要知道 HTML。

前端人员只能处理 HTML，并在任何需要 Django 信息的地方留下 HTML 注释。稍后一个后端人员会用 DTL 语法替换 HTML 注释，因此不需要 HTML 知识。

## Django 模板语言(DTL)的基本结构

DTL 语法与 Python 非常相似。它包括:

*   模板标签
*   模板变量
*   模板过滤器
*   模板注释

我们现在将逐个研究它们。

## 1.**模板标签**

模板标签执行一个功能或过程。也就是他们**“做”**某件事。模板标记语法:

```py
{% Tag %}

```

模板标签本身属于 **5** 不同的类型:

### 1.1 **条件语句**

这些，类似于 Python 中的 **[条件语句](https://www.askpython.com/python/python-if-else-elif-statement)** ，是用来执行逻辑的。

下面显示了一个示例:

```py
{% if %}
    <code>
{% end if %}

```

### **1.2 个循环**

这个，类似于 **[python 循环](https://www.askpython.com/python/python-loops-in-python)** ，用来迭代循环中的变量。

```py
{% for x in y %}
    <code>
{% endfor %}

```

### 1.3 **块声明**

块声明主要用于 [**模板继承**。](https://www.askpython.com/django/django-template-inheritance)

语法如下所示:

```py
{% block content %}
    <code>
{% endblock %}

```

### 1.4 **文件内含物**

该命令将其他 HTML 文件包含到当前文件中。

```py
{% include “header.html(file name)” %}

```

### 1.5 **文件继承**

下面的命令将其他 HTML 文件继承到当前文件中。

```py
{% extends “base.html(file name)” %}

```

## 2.**模板变量**

DTL 中的模板变量函数类似于 Python 中的[变量](https://www.askpython.com/python/python-variables)。语法:

```py
{{ <Variable_Name> }}

```

下面给出了一些模板变量示例:

*   **简单变量** : {{ title }}，{{ x }}
*   **列表属性** : {{ fruits_list.0 }}
*   **对象属性** : {{ name.title }}
*   **字典属性** : {{ dict.key }}

这些变量的数据直接从 Python 代码中提取，值可以通过使用上述语法在 HTML 代码中实现。

## 3.**模板过滤器**

模板过滤器用于对模板变量进行过滤。模板过滤器的语法:

```py
{{ <Variable_Name> | <filter_by_attribute> }}

```

一些最常用的模板过滤器示例如下:

*   **改变大小写** : {{姓名|头衔}}，{{字符|大写}}
*   **列表过滤器/切片** : {{ list|slice = " :5 " }}
*   **截断** : {{ name|truncatewords : 80 }}
*   **默认值** : {{ value|default ="0" }}

## 4.模板注释

顾名思义，这相当于 DTL 的 [python 注释](https://www.askpython.com/python/python-comments)。Templat 注释语法:

```py
{# <Comment> #}

```

就像在 python 中一样，comment 属性中的代码不会被控制台执行。

## 结论

就这样，伙计们！！这都是关于 Django 模板语言的。有了这个，你就可以高效地用 Python 链接 HTML 代码了。一定要查看关于 Django 模板语言的官方文档。

下一篇文章再见！！在那之前继续练习！！