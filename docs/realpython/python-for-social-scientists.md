# 面向社会科学家的 Python

> 原文：<https://realpython.com/python-for-social-scientists/>

Python 是社会科学家中越来越流行的数据分析工具。在许多已经成熟的库的支持下，R 和 Stata 用户越来越多地转向 Python，以便在不牺牲这些老程序多年来积累的功能的情况下利用 Python 的美丽、灵活和性能。

但是，尽管 Python 提供了很多东西，但现有的 Python 资源并不总是很适合社会科学家的需求。考虑到这一点，我最近创建了一个新资源——[www.data-analysis-in-python.org](http://www.data-analysis-in-python.org)(DAP)——专门为社会科学家 Python 用户的目标和愿望量身定制。

然而，这个网站并不是一套新的教程——世界上有比足够多的 Python 教程还要多的 T2 教程。更确切地说，该网站的目的是管理和注释现有资源，并为用户提供关注哪些主题和跳过哪些主题的指导。

## 为什么是社会科学家的网站？

社会科学家——事实上，大多数数据科学家——花费大部分时间试图将个体的、特殊的数据集整合成运行统计分析所需的形状。这使得大多数社会科学家使用 Python 的方式与大多数软件开发人员使用 Python 的方式有着根本的不同。

社会科学家主要对编写执行一系列命令(记录变量、合并数据集、解析文本文档等)的相对简单的程序(脚本)感兴趣。)将他们的数据整理成他们可以分析的形式。因为他们通常为特定的、特殊的应用程序和数据集编写脚本，他们通常不会专注于编写具有大量抽象的代码。

换句话说，社会科学家倾向于主要对学习如何有效地使用现有工具感兴趣，而不是开发新工具。

正因为如此，学习 Python 的社会科学家在技能发展方面比软件开发人员有着不同的优先权。然而，大多数在线教程是为开发人员或计算机科学学生编写的，所以 DAP 的目的之一是为社会科学家提供一些指导，让他们知道在早期培训中应该优先考虑的技能。特别是， [DAP 建议](http://www.data-analysis-in-python.org/2_basic_python.html):

**立即需要:**

*   数据类型:整数、浮点数、字符串、[布尔值](https://realpython.com/python-or-operator/)、列表、字典和集合(元组是可选的)
*   定义函数
*   写循环
*   理解可变和不可变数据类型
*   操作字符串的方法
*   导入第三方模块
*   阅读和解释错误

**你在某个时候会想知道，但不是马上就要知道的事情:**

*   高级调试工具(如 [pdb](https://realpython.com/python-debugging-pdb/)
*   文件输入/输出(您将使用的大多数库都有工具来简化这一过程)

**不需要:**

*   定义或编写类
*   了解异常

[*Remove ads*](/account/join/)

## 熊猫

今天，大多数实证社会科学仍然围绕表格数据组织，这意味着数据在每一列中以不同的变量呈现，在每一行中以不同的观察值呈现。因此，当许多使用 Python 的社会科学家在 Python 入门教程中找不到表格数据结构时，他们会感到有些困惑。为了解决这一困惑，DAP 尽最大努力尽快向用户介绍[熊猫](https://realpython.com/pandas-python-explore-dataset/)图书馆，提供[教程链接和一些关于注意](http://www.data-analysis-in-python.org/3_pandas.html)陷阱的提示。

pandas 库复制了社会科学家习惯于在 Stata 或 R 中发现的许多功能——数据可以用表格格式表示，列变量可以很容易地标记，不同类型的列(如 floats 和 strings)可以组合在同一个数据集中。

熊猫也是社会科学家可能使用的许多其他工具的门户，如图形库( [seaborn](https://seaborn.pydata.org/) 和 [ggplot2](http://ggplot.yhathq.com/) )和 [statsmodels](https://www.statsmodels.org/) 计量经济学库。

## 按研究领域分类的其他图书馆

虽然所有希望使用 Python 的社会科学家都需要理解核心语言，并且大多数人都希望熟悉`pandas`，但是 Python 生态系统充满了特定于应用程序的库，这些库只对一部分用户有用。考虑到这一点，DAP 提供了图书馆的概述，以帮助在不同主题领域工作的研究人员，以及最佳使用材料的链接，以及相关注意事项的指导:

*   [网络分析](http://www.data-analysis-in-python.org/t_igraph.html) : iGraph
*   [文本分析](http://www.data-analysis-in-python.org/t_text_analysis.html) : NLTK，如果需要的话还有 coreNLP
*   [计量经济学](http://www.data-analysis-in-python.org/t_statsmodels.html):统计模型
*   [绘图](http://www.data-analysis-in-python.org/t_seaborn.html) : ggplot 和 seaborn
*   [大数据](http://www.data-analysis-in-python.org/t_big_data.html) : dask 和 pyspark
*   [地理空间分析](http://www.data-analysis-in-python.org/t_gis.html) : arcpy 或 geopandas
*   [让代码更快](http://www.data-analysis-in-python.org/t_super_fast.html):[iPython](https://realpython.com/effective-python-environment/#python-interpreters)(用于评测)和 numba(用于 JIT 编译)

## 想参与进来吗？

这个网站是年轻的，所以我们渴望尽可能多的内容和设计的投入。如果你有这方面的经验，你想分享*请*给我发一封[的电子邮件](http://www.nickeubank.com/cv-and-contact-info/)或者在 Github 上发一条[的评论](https://github.com/nickeubank/data-analysis-in-python)。

* * *

*这是范德比尔特大学民主制度研究中心的博士后尼克·尤班克的客座博文**