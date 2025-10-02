# 浅入浅出 Flask 框架：url_for

2014-06-28

`url_for`可以让你以软编码的形式生成 url，提供开发效率。

## 示例

* * *

### 建立 Flask 项目

按照以下命令建立 Flask 项目 HelloWorld:

```py
mkdir HelloWorld
mkdir HelloWorld/static
mkdir HelloWorld/templates
touch HelloWorld/index.py 
```

在`HelloWorld/static`创建目录`uploads`，copy 一张图片放到`uploads`目录中，命名为`01.jpg`。

### 编写代码

编辑`HelloWorld/index.py`：

```py
from flask import Flask, url_for

app = Flask(__name__)

@app.route('/')
def hello_world():
    pass

@app.route('/user/<name>')
def user(name):
    pass

@app.route('/page/<int:num>')
def page(num):
    pass

@app.route('/test')
def test():
    print url_for('hello_world')
    print url_for('user', name='letian')
    print url_for('page', num=1, q='hadoop mapreduce 10%3')
    print url_for('static', filename='uploads/01.jpg')
    return ''

if __name__ == '__main__':
    app.run(debug=True) 
```

运行`HelloWorld/index.py`。然后在浏览器中访问`http://127.0.0.1:5000/test`，`HelloWorld/index.py`将输出以下信息：

```py
/
/user/letian
/page/1?q=hadoop+mapreduce+10%253
/static//uploads/01.jpg 
```