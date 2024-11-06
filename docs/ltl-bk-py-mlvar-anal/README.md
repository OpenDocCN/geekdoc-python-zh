# Python 多元分析小书

本小册子告诉您如何使用 Python 生态系统进行一些简单的多元分析，重点是主成分分析（PCA）和线性判别分析（LDA）。

jupyter notebook 可以在其[github 仓库](https://github.com/yianni/a_little_book_of_python_for_multivariate_analysis) [https://github.com/yianni/a_little_book_of_python_for_multivariate_analysis]找到。

## 笔记

本小册子假设读者具有一些多元分析的基础知识，而且小册子的主要重点不是解释多元分析，而是解释如何使用 Python 进行这些分析。

函数中的命名约定保持与原始源的一致性。变量被重命名为更通用的名称，因此如果您的数据的第一列包含数据类，则可以加载自己的数据集并运行笔记本。请参阅读取数据的单元格。

Python 代码旨在易于理解，就像原始源中的 R 代码一样，而不是具有计算和内存效率。

## 目录

+   Python 多元分析小册子

    +   设置 Python 环境

        +   安装 Python

        +   库

        +   导入库

        +   Python 控制台

    +   将多元分析数据读入 Python

    +   绘制多元数据

        +   矩阵散点图

        +   带有数据点标签的散点图

        +   概要图

    +   计算多元数据的汇总统计信息

        +   每个组的平均值和方差

        +   一个变量的组间方差和组内方差

        +   两个变量的组间协方差和组内协方差

        +   计算多元数据的相关性

        +   变量标准化

    +   主成分分析

        +   确定保留多少主成分

        +   主成分的载荷

        +   主成分的散点图

    +   线性判别分析

        +   判别函数的载荷

        +   判别函数实现的分离度

        +   LDA 值的堆叠直方图

        +   判别函数的散点图

        +   分配规则和误分类率

            +   Python 方式

    +   链接和进一步阅读

    +   致谢

    +   联系方式

    +   许可证

## 许可证

[![知识共享许可证](https://i.creativecommons.org/l/by-sa/4.0/88x31.png)](http://creativecommons.org/licenses/by-sa/4.0/)

《Python 多元分析小书》由 Yiannis Gatsoulis 根据[知识共享署名-相同方式共享 4.0 国际许可协议](http://creativecommons.org/licenses/by-sa/4.0/)许可。

基于 Avril Coghlan 的 [《R 多元分析小册子》](https://little-book-of-r-for-multivariate-analysis.readthedocs.org/en/latest/src/multivariateanalysis.html) 的作品，使用 [CC-BY-3.0](http://creativecommons.org/licenses/by/3.0/) 许可。© 2016 版权所有，Yiannis Gatsoulis 创作。使用 [Sphinx](http://sphinx-doc.org/) 1.3.4 创建。
