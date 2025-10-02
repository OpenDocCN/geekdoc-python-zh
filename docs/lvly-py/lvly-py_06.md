# ![rsticon](img/rst.png) 图书工程说明

运用 [基于 Sphinx 的图书协同](http://code.google.com/p/openbookproject/wiki/FlowSphinx) [http://code.google.com/p/openbookproject/wiki/FlowSphinx] 流程和方式进行

## Shinx 使用

参考:

> Sphinx 快速开始 - pymotwcn
> 
> *   [`code.google.com/p/pymotwcn/wiki/SphinxprojectHowto`](http://code.google.com/p/pymotwcn/wiki/SphinxprojectHowto)

## 代码引用

正文内置::

@route('%s/'%ini.urlprefix) def index():

> __urlog("INFO","idx++") return template('index.tpl',urlprefix=ini.urlprefix)

外部包含:

```py
@route('%s/'%ini.urlprefix)
def index():
    __urlog("INFO","idx++")
    return template('index.tpl',urlprefix=ini.urlprefix)

```

## 表格线性声明

中文的非等宽性导致这种字符艺术式的图表很难作！ 所以,使用列表也可以方便的生成表格:

```py
.. list-table:: 实例
   :widths: 15 10 30
   :header-rows: 1

   * - Treat
     - Quantity
     - Description
   * - Albatross
     - 2.99
     - On a stick!
   * - Crunchy Frog
     - 1.49
     - If we took the bones out, it wouldn't be
       crunchy, now would it?
   * - Gannet Ripple
     - 1.99
     - On a stick!
```

**效果**

实例

| Treat | Quantity | Description |
| --- | --- | --- |
| Albatross | 2.99 | On a stick! |
| Crunchy Frog | 1.49 | If we took the bones out, it wouldn't be crunchy, now would it? |
| Gannet Ripple | 1.99 | On a stick! |

## 段落层次约定

使用 [reSTsections](http://sphinx.pocoo.org/rest.html#sections) [http://sphinx.pocoo.org/rest.html#sections]

```py
共分 4 级
大标题
=======================

小标题
-----------------------

二级标题
^^^^^^^^^^^^^^^^^^^^^^^

三级标题
"""""""""""""""""""""""
```

再小，就使用列表!:

> *   列表项目 1
> *   列表项目 2
> *   ...

**效果:**

# 大标题

## 小标题

### 二级标题

#### 三级标题

© Copyright 2010, Zoom.Quiet. Last updated on 130306 09:15. Created using [Sphinx](http://sphinx.pocoo.org/) 1.1.3.

Brought to you by Read the Docs

*   latest