# 使用 pywhois 检索 whois 信息

> 原文：<https://www.pythonforbeginners.com/python-on-the-web/using-pywhois>

## pywhois 是什么？

pywhois 是一个 Python 模块，用于检索域名的 whois 信息。pywhois 与 Python 2.4+一起工作，没有外部依赖[【源】](https://code.google.com/p/pywhois/ "pywhois")

## 装置

pywhois 的安装是通过 pip 命令完成的。

```py
pip install python-whois 
```

现在，当软件包安装完成后，您就可以开始使用它了。请记住，您必须首先导入它。

```py
import whois
```

## pywhois 用法

我们可以使用 pywhois 模块直接查询 whois 服务器，并解析给定域的 WHOIS 数据。我们能够提取所有流行顶级域名(com、org、net……)的数据

## pywhois 示例

在 [pywhois](https://code.google.com/p/pywhois/ "pywhois_projects") 项目网站上，我们可以看到如何使用 pywhois 提取数据。

让我们从导入 whois 模块开始，并创建一个变量。

```py
>>> import whois >>> w = whois.whois('pythonforbeginners.com’)
```

要打印所有找到的属性的值，我们只需输入:

```py
>>> print w 
```

输出应该如下所示:

```py
creation_date: [datetime.datetime(2012, 9, 15, 0, 0), '15 Sep 2012 20:41:00']
domain_name: ['PYTHONFORBEGINNERS.COM', 'pythonforbeginners.com']
...
...
updated_date: 2013-08-20 00:00:00
whois_server: whois.enom.com 
```

我们可以打印出任何我们想要的属性。假设您只想打印出截止日期:

```py
>>> w.expiration_date 
```

显示从 whois 服务器下载的内容:

```py
>>> w.text 
```

为了使程序更具交互性，我们可以添加一个提示，用户可以在其中输入他们想要检索的 WHOIS 信息。

```py
import whois

data = raw_input("Enter a domain: ")
w = whois.whois(data)

print w 
```

在 pywhois 模块的帮助下，我们可以使用 Python 进行 whois 查找。

更多阅读

[http://code.google.com/p/pywhois/](https://code.google.com/p/pywhois/ "pywhois_module")