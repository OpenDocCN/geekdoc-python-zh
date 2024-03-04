# Python 机械化备忘单

> 原文：<https://www.pythonforbeginners.com/cheatsheet/python-mechanize-cheat-sheet>

## 使机械化

在 web 表单中导航的一个非常有用的 python 模块是 Mechanize。在之前的一篇文章中，我写了关于[“用机械化在 Python 中浏览”](https://www.pythonforbeginners.com/python-on-the-web/browsing-in-python-with-mechanize "browingMechanize")。今天我在 [scraperwiki](https://views.scraperwiki.com/run/python_mechanize_cheat_sheet/ "scraperwiki") 上发现了这个优秀的小抄，我想和大家分享一下。

## 创建浏览器对象

创建一个浏览器对象，并给它一些可选的设置。

```py
import mechanize
br = mechanize.Browser()
br.set_all_readonly(False)    # allow everything to be written to
br.set_handle_robots(False)   # ignore robots
br.set_handle_refresh(False)  # can sometimes hang without this
br.addheaders =   	      	# [('User-agent', 'Firefox')] 
```

## 打开网页

打开网页并检查其内容

```py
response = br.open(url)
print response.read()      # the text of the page
response1 = br.response()  # get the response again
print response1.read()     # can apply lxml.html.fromstring() 
```

## 使用表单

列出页面中的表单

```py
for form in br.forms():
    print "Form name:", form.name
    print form 
```

要继续，mechanize 浏览器对象必须选择一个表单

```py
br.select_form("form1")         # works when form has a name
br.form = list(br.forms())[0]  # use when form is unnamed 
```

## 使用控件

循环访问窗体中的控件。

```py
for control in br.form.controls:
    print control
    print "type=%s, name=%s value=%s" % (control.type, control.name, br[control.name]) 
```

可以通过名称找到控件

```py
control = br.form.find_control("controlname") 
```

拥有一个选择控件可以告诉你可以选择哪些值

```py
if control.type == "select":  # means it's class ClientForm.SelectControl
    for item in control.items:
    print " name=%s values=%s" % (item.name, str([label.text  for label in item.get_labels()])) 
```

因为“Select”类型的控件可以有多个选择，所以它们必须用列表设置，即使它是一个元素。

```py
print control.value
print control  # selected value is starred
control.value = ["ItemName"]
print control
br[control.name] = ["ItemName"]  # equivalent and more normal 
```

文本控件可以设置为字符串

```py
if control.type == "text":  # means it's class ClientForm.TextControl
    control.value = "stuff here"
br["controlname"] = "stuff here"  # equivalent 
```

控件可以设置为只读和禁用。

```py
control.readonly = False
control.disabled = True 
```

或者像这样禁用它们

```py
for control in br.form.controls:
   if control.type == "submit":
       control.disabled = True 
```

## 提交表单

完成表格后，您可以提交

```py
 response = br.submit()
print response.read()
br.back()   # go back 
```

## 查找链接

在 mechanize 中跟踪链接很麻烦，因为您需要有 link 对象。有时候把它们都搞定，从文本中找到你想要的链接会更容易。

```py
for link in br.links():
    print link.text, link.url 
```

跟随链接并点击链接等同于提交并点击

```py
request = br.click_link(link)
response = br.follow_link(link)
print response.geturl() 
```

我希望您对 Python 中的 Mechanize 模块有了更多的了解。