## 关联

尽量保持任何贡献与 本指南目的 相关。

*   避免在主题中包含太多与 Python 开发并不直接相关的信息。
*   如果其他资源已存在，最好以链接的形式来展示。确保描述出你所链接的内容和原因。
*   [Cite](http://sphinx.pocoo.org/rest.html?highlight=citations#citations) [http://sphinx.pocoo.org/rest.html?highlight=citations#citations] 引用在需要的地方。
*   如果某主题并不与 Python 直接相关，但是和 Python 之间的关联又很有用（比如 Git、GitHub、数据库等），以链接的形式引用这些资源，并描述为什么它对 Python 有用。
*   如果疑问，就去询问。

## 标题

使用下列风格作为标题。

章节标题：

```py
#########
章节 1
######### 
```

页面标题：

```py
===================
时间是种幻觉
=================== 
```

小节标题:

```py
午餐时间加倍
------------------- 
```

次小节标题:

```py
非常深
~~~~~~~~~ 
```

## 换行

每 78 个字符进行文字换行。必要时可以超过 78 个字符，尤其是那种换行使得源内容更难阅读的情况。

## 代码例子

所有代码示例要在 70 个字符进行换行，以避免出现水平滚动条。

命令行例子：

```py
.. code-block:: console

    $ run command --help
    $ ls .. 
```

确保每行前面包含了 `$` 前缀。

Python 解释器例子：

```py
Label the example::

.. code-block:: python

    >>> import this 
```

Python 例子：

```py
Descriptive title::

.. code-block:: python

    def get_answer():
        return 42 
```

## 外部链接

*   链接时最好使用众所周知的主题（比如一些合适的名词）：

    ```py
    Sphinx_ 通常用来文档化 Python。

    .. _Sphinx: http://sphinx.pocoo.org 
    ```

*   最好使用带有内联链接的描述性标签，而不是单纯的链接:

    ```py
    阅读 `Sphinx 教程 <http://sphinx.pocoo.org/tutorial.html>`_ 
    ```

*   避免使用诸如“点击这里”、“这个”等标签。最好使用描述性标签（值得搜索引擎优化，SEO worthy）。

## 指向指南内部章节的链接

要交叉引用本文档的其他部分，使用 [:ref:](http://sphinx.pocoo.org/markup/inline.html#cross-referencing-arbitrary-locations) [http://sphinx.pocoo.org/markup/inline.html#cross-referencing-arbitrary-locations] 关键字和标签。

要使引用标签更加清晰和独特，通常加上一个 `-ref` 后缀：

```py
.. _some-section-ref:

Some Section
------------ 
```

## 注意和警告

使用适当的 [警告指示](http://sphinx.pocoo.org/rest.html#directives) [http://sphinx.pocoo.org/rest.html#directives] 来说明注意内容。

注意:

```py
.. note::
    The Hitchhiker’s Guide to the Galaxy has a few things to say
    on the subject of towels. A towel, it says, is about the most
    massively useful thing an interstellar hitch hiker can have. 
```

警告:

```py
.. warning:: DON'T PANIC 
```

## 要做的事

请用 [todo 指示](http://sphinx.pocoo.org/ext/todo.html?highlight=todo#directive-todo) [http://sphinx.pocoo.org/ext/todo.html?highlight=todo#directive-todo] 来标记本指南中任何未完成的部分。避免使 要做的事 混乱，为未完的文档或者大量未完的小节使用单独的 `todo`。

```py
.. todo::
    Learn the Ultimate Answer to the Ultimate Question
    of Life, The Universe, and Everything 
``` © 版权所有 2014\. A <a href="http://kennethreitz.com/pages/open-projects.html">Kenneth Reitz</a> 工程。 <a href="http://creativecommons.org/licenses/by-nc-sa/3.0/"> Creative Commons Share-Alike 3.0</a>.

# 索引

**P** | **符号**

## P

|  
PATH, [[1]](starting/install/win.html#index-0), [[2]](starting/install/win.html#index-1), [[3]](starting/install/win.html#index-2)

 |  
Python 提高建议

PEP 1

PEP 20, [[1]](writing/style.html#index-1)

PEP 249

PEP 257

PEP 282

PEP 3101

PEP 3132

PEP 3333

PEP 391

PEP 8, [[1]](intro/community.html#index-1), [[2]](intro/learning.html#index-0), [[3]](writing/style.html#index-2)

PEP 8#comments

 |

## 符号

|  
环境变量

PATH, [[1]](starting/install/win.html#index-0), [[2]](starting/install/win.html#index-1), [[3]](starting/install/win.html#index-2)

 |

© 版权所有 2014\. A <a href="http://kennethreitz.com/pages/open-projects.html">Kenneth Reitz</a> 工程。 <a href="http://creativecommons.org/licenses/by-nc-sa/3.0/"> Creative Commons Share-Alike 3.0</a>.