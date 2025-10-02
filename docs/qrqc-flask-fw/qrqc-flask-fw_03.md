# 浅入浅出 Flask 框架：处理客户端通过 POST 方法传送的数据

2014-6-24

作为一种 HTTP 请求方法，POST 用于向指定的资源提交要被处理的数据。我们在某网站注册用户、写文章等时候，需要将数据保存在服务器中，这是一般使用 POST 方法。

本文使用 python 的 requests 库模拟客户端。

## 建立 Flask 项目

* * *

按照以下命令建立 Flask 项目 HelloWorld:

```py
mkdir HelloWorld
mkdir HelloWorld/static
mkdir HelloWorld/templates
touch HelloWorld/index.py 
```

## 简单的 POST

* * *

以用户注册为例子，我们需要向服务器`/register`传送用户名`name`和密码`password`。如下编写`HelloWorld/index.py`。

```py
from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'hello world'

@app.route('/register', methods=['POST'])
def register():
    print request.headers
    print request.form
    print request.form['name']
    print request.form.get('name')
    print request.form.getlist('name')
    print request.form.get('nickname', default='little apple')
    return 'welcome'

if __name__ == '__main__':
    app.run(debug=True) 
```

`@app.route('/register', methods=['POST'])`是指 url`/register`只接受 POST 方法。也可以根据需要修改`methods`参数，例如

```py
@app.route('/register', methods=['GET', 'POST'])  # 接受 GET 和 POST 方法 
```

具体请参考[http-methods](http://flask.pocoo.org/docs/quickstart/#http-methods)。

客户端`client.py`内容如下：

```py
import requests

user_info = {'name': 'letian', 'password': '123'}
r = requests.post("http://127.0.0.1:5000/register", data=user_info)

print r.text 
```

运行`HelloWorld/index.py`，然后运行`client.py`。`client.py`将输出：

```py
welcome 
```

而`HelloWorld/index.py`在终端中输出以下调试信息（通过`print`输出）：

```py
Content-Length: 24
User-Agent: python-requests/2.2.1 CPython/2.7.6 Windows/8
Host: 127.0.0.1:5000
Accept: */*
Content-Type: application/x-www-form-urlencoded
Accept-Encoding: gzip, deflate, compress

 ImmutableMultiDict([('password', u'123'), ('name', u'letian')])
letian
letian
[u'letian']
little apple 
```

前 6 行是 client.py 生成的 HTTP 请求头，由于`print request.headers`输出。

`print request.form`的结果是：

```py
ImmutableMultiDict([('password', u'123'), ('name', u'letian')]) 
```

这是一个`ImmutableMultiDict`对象。关于`request.form`，更多内容请参考[flask.Request.form](http://flask.pocoo.org/docs/api/?highlight=request.form#flask.Request.form)。关于`ImmutableMultiDict`，更多内容请参考[werkzeug.datastructures.MultiDict](http://werkzeug.pocoo.org/docs/datastructures/#werkzeug.datastructures.MultiDict)。

`request.form['name']`和`request.form.get('name')`都可以获取`name`对应的值。对于`request.form.get()`可以为参数`default`指定值以作为默认值。所以：

```py
print request.form.get('nickname', default='little apple') 
```

输出的是默认值

```py
little apple 
```

如果`name`有多个值，可以使用`request.form.getlist('name')`，该方法将返回一个列表。我们将 client.py 改一下：

```py
import requests

user_info = {'name': ['letian', 'letian2'], 'password': '123'}
r = requests.post("http://127.0.0.1:5000/register", data=user_info)

print r.text 
```

此时运行`client.py`，`print request.form.getlist('name')`将输出：

```py
[u'letian', u'letian2'] 
```

## 上传文件

* * *

这一部分的代码参考自[How to upload a file to the server in Flask](http://runnable.com/UiPcaBXaxGNYAAAL/how-to-upload-a-file-to-the-server-in-flask-for-python)。

假设将上传的图片只允许’png’、’jpg’、’jpeg’、’git’这四种格式，通过 url`/upload`使用 POST 上传，上传的图片存放在服务器端的`static/uploads`目录下。

首先在项目`HelloWorld`中创建目录`static/uploads`：

```py
$ mkdir HelloWorld/static/uploads 
```

`werkzeug`库可以判断文件名是否安全，例如防止文件名是`../../../a.png`，安装这个库：

```py
$ pip install werkzeug 
```

修改`HelloWorld/index.py`：

```py
from flask import Flask, request
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg', 'gif'])

# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def hello_world():
    return 'hello world'

@app.route('/upload', methods=['POST'])
def upload():
    upload_file = request.files['image01']
    if upload_file and allowed_file(upload_file.filename):
        filename = secure_filename(upload_file.filename)
        upload_file.save(os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], filename))
        return 'hello, '+request.form.get('name', 'little apple')+'. success'
    else:
        return 'hello, '+request.form.get('name', 'little apple')+'. failed'

if __name__ == '__main__':
    app.run(debug=True) 
```

`app.config`中的 config 是字典的子类，可以用来设置自有的配置信息，也可以设置自己的配置信息。函数`allowed_file(filename)`用来判断`filename`是否有后缀以及后缀是否在`app.config['ALLOWED_EXTENSIONS']`中。

客户端上传的图片必须以`image01`标识。`upload_file`是上传文件对应的对象。`app.root_path`获取`index.py`所在目录在文件系统中的绝对路径。`upload_file.save(path)`用来将`upload_file`保存在服务器的文件系统中，参数最好是绝对路径，否则会报错（网上很多代码都是使用相对路径，但是笔者在使用相对路径时总是报错，说找不到路径）。函数`os.path.join()`用来将使用合适的路径分隔符将路径组合起来。

好了，定制客户端`client.py`：

```py
import requests

files = {'image01': open('01.jpg', 'rb')}
user_info = {'name': 'letian'}
r = requests.post("http://127.0.0.1:5000/upload", data=user_info, files=files)

print r.text 
```

当前目录下的`01.jpg`将上传到服务器。运行`client.py`，结果如下：

```py
hello, letian. success 
```

然后，我们可以在`static/uploads`中看到文件`01.jpg`。

要控制上产文件的大小，可以设置请求实体的大小，例如：

```py
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 #16MB 
```

不过，在处理上传文件时候，需要使用`try:...except:...`。

如果要获取上传文件的内容可以：

```py
file_content = request.files['image01'].stream.read() 
```

## 处理 JSON

* * *

处理 JSON 时，要把请求头和响应头的`Content-Type`设置为`application/json`。

修改`HelloWorld/index.py`：

```py
from flask import Flask, request, Response
import json

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'hello world'

@app.route('/json', methods=['POST'])
def my_json():
    print request.headers
    print request.json
    rt = {'info':'hello '+request.json['name']}
    return Response(json.dumps(rt),  mimetype='application/json')

if __name__ == '__main__':
    app.run(debug=True) 
```

修改后运行。

修改`client.py`：

```py
import requests, json

user_info = {'name': 'letian'}
headers = {'content-type': 'application/json'}
r = requests.post("http://127.0.0.1:5000/json", data=json.dumps(user_info), headers=headers)
print r.headers
print r.json() 
```

运行`client.py`，将显示：

```py
CaseInsensitiveDict({'date': 'Tue, 24 Jun 2014 12:10:51 GMT', 'content-length': '24', 'content-type': 'application/json', 'server': 'Werkzeug/0.9.6 Python/2.7.6'})
{u'info': u'hello letian'} 
```

而`HelloWorld/index.py`的调试信息为：

```py
Content-Length: 18
User-Agent: python-requests/2.2.1 CPython/2.7.6 Windows/8
Host: 127.0.0.1:5000
Accept: */*
Content-Type: application/json
Accept-Encoding: gzip, deflate, compress

 {u'name': u'letian'} 
```

这个比较简单，就不多说了。另外，如果需要响应头具有更好的可定制性，可以如下修改`my_json()`函数：

```py
@app.route('/json', methods=['POST'])
def my_json():
    print request.headers
    print request.json
    rt = {'info':'hello '+request.json['name']}
    response = Response(json.dumps(rt),  mimetype='application/json')
    response.headers.add('Server', 'python flask')
    return response 
```