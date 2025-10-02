# JSON

[json](https://docs.python.org/2/library/json.html) [https://docs.python.org/2/library/json.html] 库可以自字符串或文件中解析 JSON。 该库解析 JSON 后将其转为 Python 字典或者列表。它也可以转换 Python 字典或列表为 JSON 字符串。

## 解析 JSON

创建下面包含 JSON 数据的字符串

```py
json_string = '{"first_name": "Guido", "last_name":"Rossum"}' 
```

它可以被这样解析：

```py
import json
parsed_json = json.loads(json_string) 
```

然后它可以像一个常规的字典那样使用:

```py
print(parsed_json['first_name'])
"Guido" 
```

你可以把下面这个对象转为 JSON：

```py
d = {
    'first_name': 'Guido',
    'second_name': 'Rossum',
    'titles': ['BDFL', 'Developer'],
}

print(json.dumps(d))
'{"first_name": "Guido", "last_name": "Rossum", "titles": ["BDFL", "Developer"]}' 
```

## simplejson

JSON 库是 Python2.6 版中加入的。如果你使用更早版本的 Python， 可以通过 PyPI 获取 [simplejson](https://simplejson.readthedocs.org/en/latest/) [https://simplejson.readthedocs.org/en/latest/] 库。

simplejson 类似 json 标准库，它使得使用老版本 Python 的开发者们可以使用 json 库中的最新特性。

如果 json 库不可用，你可以将 simplejson 取别名为 json 来使用：

```py
import simplejson as json 
```

在将 simplejson 当成 json 导入后，上面的例子会像你在使用标准 json 库一样正常运行。

© 版权所有 2014\. A <a href="http://kennethreitz.com/pages/open-projects.html">Kenneth Reitz</a> 工程。 <a href="http://creativecommons.org/licenses/by-nc-sa/3.0/"> Creative Commons Share-Alike 3.0</a>.