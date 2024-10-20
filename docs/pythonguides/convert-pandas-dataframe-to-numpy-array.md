# 如何在 Python 中将 Pandas 数据帧转换成 NumPy 数组

> 原文：<https://pythonguides.com/convert-pandas-dataframe-to-numpy-array/>

[![Python & Machine Learning training courses](img/49ec9c6da89a04c9f45bab643f8c765c.png)](https://sharepointsky.teachable.com/p/python-and-machine-learning-training-course)

在这个 [Python 熊猫教程](https://pythonguides.com/pandas-in-python/)中，我们将学习如何在 Python 中**转换熊猫数据帧为 NumPy 数组**。此外，我们将涵盖这些主题。

*   将 Pandas 数据帧转换为不带标题的 NumPy 数组
*   将 Pandas 数据帧转换为不带索引的 NumPy 数组
*   将 Pandas 数据帧转换为 NumPy 数组
*   将 Pandas 系列转换为 NumPy 数组
*   将 Pandas 数据帧转换为 3d 数字数组
*   将 Pandas 数据帧转换为 2d 数字数组
*   将 Pandas 数据帧转换为 NumPy 矩阵

我们使用了从 [Kaggle](https://www.kaggle.com/jackogozaly/steam-player-data) 下载的 [Steam 播放器数据](https://www.kaggle.com/jackogozaly/steam-player-data)。

目录

[](#)

*   [将 Pandas 数据帧转换为不带标题的 NumPy 数组](#Convert_Pandas_DataFrame_to_NumPy_Array_Without_Header "Convert Pandas DataFrame to NumPy Array Without Header")
*   [将 Pandas 数据帧转换为不带索引的 NumPy 数组](#Convert_Pandas_DataFrame_to_NumPy_Array_Without_Index "Convert Pandas DataFrame to NumPy Array Without Index")
*   [将 Pandas 数据帧转换为 NumPy 数组](#Convert_Pandas_DataFrame_to_NumPy_Array "Convert Pandas DataFrame to NumPy Array")
*   [将 Pandas 系列转换为 NumPy 数组](#Convert_Pandas_Series_to_NumPy_Array "Convert Pandas Series to NumPy Array")
*   [将 Pandas 数据帧转换为 3D NumPy 数组](#Convert_Pandas_DataFrame_to_3D_NumPy_Array "Convert Pandas DataFrame to 3D NumPy Array")
*   [将熊猫数据帧转换为 2D 数字数组](#Convert_Pandas_DataFrame_to_2D_NumPy_Array "Convert Pandas DataFrame to 2D NumPy Array")
*   [将熊猫数据帧转换为 NumPy 矩阵](#Convert_Pandas_DataFrame_to_NumPy_Matrix "Convert Pandas DataFrame to NumPy Matrix")

## 将 Pandas 数据帧转换为不带标题的 NumPy 数组

在这一节中，我们将学习如何用 Python 将 pandas dataframe 转换成不带头文件的 Numpy 数组。

*   使用 `dataframe.to_numpy()` 方法，我们可以将任何数据帧转换为 numpy 数组。
*   默认情况下，在此方法之后生成的 Numpy 数组没有标头。
*   虽然标头不可见，但可以通过引用数组名来调用它。在这种情况下，数组名称将是列名，如“月 _ 年”、“收益”、“URL”等。
*   下面是在 Jupyter 笔记本上的实现。