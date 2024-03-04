# 使用 Python 的 Bitly Shortener

> 原文：<https://www.pythonforbeginners.com/code-snippets-source-code/bitly-shortener-with-python>

## 开始之前

Bitly 允许用户缩短、分享和跟踪链接(URL)。

这是一种保存、分享和发现网络链接的方式。

Bitly 提供了公共 API，目的是让 python 程序员更容易使用。

## 第一步

第一步是前往[dev.bitly.com](https://dev.bitly.com "dev.bitly.com")，在那里你会找到 API
文档、最佳实践、代码库、公共数据集。

## 什么是 API 密钥？

互联网上的许多服务(如 Twitter、脸书..)要求你
有一个“API 密匙”。

[应用编程接口](https://en.wikipedia.org/wiki/Application_programming_interface_key "wikipedia_api")密钥(API key)是由调用 API 的
计算机程序传入的代码，用于向网站标识调用程序、其开发者、
或其用户。

API 密钥用于跟踪和控制 API 的使用方式，例如
防止恶意使用或滥用 API。

API 密钥通常既作为唯一的标识符，又作为用于
认证的秘密令牌，并且通常具有一组对与之相关联的 API
的访问权限。

## 获取 Bitly API 密钥

为了能够让我们缩短链接(见下文)，我们必须注册一个 API 键。

注册程序很容易解释，所以我不会在这篇文章中涉及。

在此创建您的 Bitly API 密钥

## 比特代码库

许多开发人员已经用几种不同的语言编写了代码库来与 bitly
API 交互。由于我们是用 Python 编程的，
我们当然对 Python 库感兴趣。

目前有三个不同的库可供选择，你可以在这里找到

在这篇文章中，我们将使用“ [bitly-api-python](https://github.com/bitly/bitly-api-python "bitly-api-python") 库，它也是
官方的 python 客户端。

## Bitly API Python

bit.ly api 的安装非常简单

```py
 # Installation using PIP

pip install bitly_api
Downloading/unpacking bitly-api
Downloading bitly_api-0.2.tar.gz
Running setup.py egg_info for package bitly-api
Installing collected packages: bitly-api
Running setup.py install for bitly-api
Successfully installed bitly-api
Cleaning up... 
```

## 缩短 URL

我们想写一个脚本，将减少 URL 长度，使分享
更容易。打开你最喜欢的文本编辑器，输入下面的代码。

将文件另存为 shortener.py

```py
#!/usr/bin/env python

# Import the modules

import bitlyapi
import sys

# Define your API information

API_USER = "your_api_username"
API_KEY = "your_api_key"

b = bitlyapi.BitLy(API_USER, API_KEY)

# Define how to use the program

usage = """Usage: python shortener.py [url]
e.g python shortener.py http://www.google.com"""

if len(sys.argv) != 2:
    print usage
    sys.exit(0)

longurl = sys.argv[1]

response = b.shorten(longUrl=longurl)

print response['url'] 
```

## Shortener.py 解释道

我们用#开始了这个计划！/usr/bin/env python

```py
#!/usr/bin/env python 
```

导入我们将在程序中使用的模块

```py
import bitlyapi
import sys 
```

定义我们的 API 信息

```py
API_USER = "your_api_username"
API_KEY = "your_api_key"
b = bitlyapi.BitLy(API_USER, API_KEY) 
```

定义如何使用程序

```py
usage = """Usage: python shortener.py [url]
e.g python shortener.py http://www.google.com"""

if len(sys.argv) != 2:
    print usage
    sys.exit(0) 
```

创建一个变量 longurl，并将值设置为传入的参数

```py
longurl = sys.argv[1] 
```

为 Bitly API 提供 longUrl response = b . shorten(longUrl = longUrl)

打印出 URL 值

```py
print response['url'] 
```