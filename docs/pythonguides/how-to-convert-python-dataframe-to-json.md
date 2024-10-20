# 如何将 Python 数据帧转换成 JSON

> 原文：<https://pythonguides.com/how-to-convert-python-dataframe-to-json/>

[![Python & Machine Learning training courses](img/49ec9c6da89a04c9f45bab643f8c765c.png)](https://sharepointsky.teachable.com/p/python-and-machine-learning-training-course)

在本 [Python 教程](https://pythonguides.com/python-hello-world-program/)中，我们将学习**如何将 Python 数据帧转换为 JSON 文件**。此外，我们将涵盖这些主题。

*   Python 数据帧到 JSON
*   Python 数据帧到 JSON 对象
*   Python 数据帧到不带索引的 JSON
*   Python 数据帧到 JSON 数组
*   Python 数据帧到 JSON 字符串
*   Python 数据帧到 JSON 格式
*   Python 数据帧到 JSON 列表
*   Python 数据帧 to_json Orient
*   带索引的 JSON 的 Python 数据帧

目录

[](#)

*   [Python 数据帧到 JSON](#Python_DataFrame_to_JSON "Python DataFrame to JSON")
*   [Python 数据帧到 JSON 对象](#Python_DataFrame_to_JSON_Object "Python DataFrame to JSON Object")
*   [Python 数据帧到不带索引的 JSON】](#Python_DataFrame_to_JSON_Without_Index "Python DataFrame to JSON Without Index")
*   [Python 数据帧到 JSON 数组](#Python_DataFrame_to_JSON_Array "Python DataFrame to JSON Array")
*   [Python 数据帧到 JSON 字符串](#Python_DataFrame_to_JSON_String "Python DataFrame to JSON String")
*   [Python 数据帧到 JSON 格式](#Python_DataFrame_to_JSON_Format "Python DataFrame to JSON Format")
*   [Python 数据帧到 JSON 列表](#Python_DataFrame_to_JSON_List "Python DataFrame to JSON List")
*   [Python data frame to _ JSON Orient](#Python_DataFrame_to_json_Orient "Python DataFrame to_json Orient")
*   [Python 数据帧到索引为](#Python_DataFrame_to_JSON_with_Index "Python DataFrame to JSON with Index")的 JSON

## Python 数据帧到 JSON

在这一节中，我们将学习如何**将 Python 数据帧转换成 JSON 文件**。熊猫数据帧可以使用 **`dataframe.to_json()`** 方法转换成 JSON 文件。

```py
DataFrame.to_json(
    path_or_buf=None, 
    orient=None, 
    date_format=None, 
    double_precision=10, 
    force_ascii=True, 
    date_unit='ms', 
    default_handler=None, 
    lines=False, 
    compression='infer', 
    index=True, 
    indent=None, 
    storage_options=None
)
```

**Python Pandas 中 dataframe.to_json()方法的参数**

| 参数 | 描述 |
| --- | --- |
| 路径缓冲区 | 提供要保存文件的文件路径和文件名。 |
| 东方 | –它接受一个字符串，您可以从给定的选项
中选择–对于系列，有 4 个选项是{'split '，' records '，' index '，' table ' }
–在这些选项中，“index”是默认值。
–对于 DataFrame，选项有 6 个选项{'split '，' records '，' index '，' columns '，' values '，' table ' }
–在这些选项中，“columns”是默认选项。 |
| 日期格式 | 日期转换的类型。可用选项有{无，'纪元'，' iso ' }
–纪元' =纪元毫秒，' iso' = ISO8601。
–默认取决于方向。
–对于 orient='table '，默认为' iso '。
–对于所有其他方向，默认为“纪元”。 |
| 双精度 | 接受 int，默认值为 10。对浮点值进行编码时使用的小数位数。 |
| force_ascii | 接受布尔值，默认值为 True。强制编码字符串为 ASCII。 |
| 日期单位 | –默认值为“ms”(毫秒)
–编码的时间单位决定时间戳和 ISO8601 精度。分别代表秒、毫秒、微秒和纳秒的' s '、' ms '、' us '、' ns '之一。 |
| 默认处理程序 | –默认值为 None
–如果对象无法转换为适合 JSON 的格式时要调用的处理程序。应该接收一个参数，该参数是要转换的对象，并返回一个可序列化的对象。 |
| 线 | –接受布尔值，默认为 False
–如果‘orient’是‘records ’,则写出行分隔的 JSON 格式。–如果不正确的“orient ”,它将抛出 ValueError，因为其他的不像列表。 |
| 压缩 | –可用选项有{'infer '，' gzip '，' bz2 '，' zip '，' xz '，None }
–表示要在输出文件中使用的压缩的字符串，仅在第一个参数是文件名时使用。默认情况下，压缩是从文件名推断出来的。 |
| 指数 | –接受布尔值，默认值为 True
–如果不想在 JSON 字符串中包含索引值，请设置 index=False。
–请注意，仅当 orient 为“分割”或“表格”时，才支持 index=False |
| 缩进 | –接受整数值
–用于缩进每条记录的空白长度。 |
| 存储选项 | –接受字典
–该选项用于存储主机、端口、用户名、密码等的连接。
–该选项主要用于 AWS S3 等云服务提供商。键值对被转发给 sffpec。 |

Parameters of df.to_json() method

*   在我们在 Jupyter Notebook 上的实现中，我们演示了必要参数的使用。
*   虽然我们已经展示了几乎所有参数的用法，但只有 `path_or_buf` 和 `orient` 是必需的，其余都是可选的。
*   这是 Jupyter 笔记本上的实现，请阅读行内注释以理解每个步骤。