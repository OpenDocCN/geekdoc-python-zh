# Python : OS.listdir 和 endswith()

> 原文：<https://www.pythonforbeginners.com/code-snippets-source-code/python-os-listdir-and-endswith>

这个简短的脚本使用 os.listdir 函数(属于 os 模块)来搜索给定的路径(".)用于所有以“”结尾的文件。txt”。

当 for 循环找到一个匹配时，它使用 append 函数将它添加到列表“newlist”中。

##### 查找所有以结尾的文件。文本文件（textfile）

```py
import os
items = os.listdir(".")

newlist = []
for names in items:
    if names.endswith(".txt"):
        newlist.append(names)
print newlist
```