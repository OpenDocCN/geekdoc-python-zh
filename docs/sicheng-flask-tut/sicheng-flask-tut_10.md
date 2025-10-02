# Flask 进阶系列(五)–文件和流

当我们要往客户端发送大量的数据，比如一个大文件时，将它保存在内存中再一次性发到客户端开销很大。比较好的方式是使用流，本篇就要介绍怎么在 Flask 中通过流的方式来将响应内容发送给客户端。此外，我们还会演示如何实现文件的上传功能，以及如何获取上传后的文件。

### 系列文章

*   Flask 进阶系列(一)–上下文环境
*   Flask 进阶系列(二)–信号
*   Flask 进阶系列(三)–Jinja2 模板引擎
*   Flask 进阶系列(四)–视图
*   Flask 进阶系列(五)–文件和流
*   Flask 进阶系列(六)–蓝图(Blueprint)
*   Flask 进阶系列(七)–应用最佳实践
*   Flask 进阶系列(八)–部署和分发
*   Flask 进阶系列(九)–测试

### 响应流的生成

Flask 响应流的实现原理就是通过 Python 的生成器，也就是大家所熟知的 yield 的表达式，将 yield 的内容直接发送到客户端。下面就是一个简单的实现：

```py
from flask import Flask, Response

app = Flask(__name__)

@app.route('/large.csv')
def generate_large_csv():
    def generate():
        for row in range(50000):
            line = []
            for col in range(500):
                line.append(str(col))

            if row % 1000 == 0:
                print 'row: %d' % row
            yield ','.join(line) + '\n'

    return Response(generate(), mimetype='text/csv')

```

这段代码会生成一个 5 万行 100M 的 csv 文件，每一行会通过 yield 表达式分别发送给客户端。运行时你会发现文件行的生成与浏览器文件的下载是同时进行的，而不是文件全部生成完毕后再开始下载。这里我们用到了响应类”flask.Response”，它是”werkzeug.wrappers.Response”类的一个包装，它的初始化方法第一个参数就是我们定义的生成器函数，第二个参数指定了响应类型。

我们将上述方法应用到模板中，如果模板的内容很大，怎么采用流的方式呢？这里我们要自己写个流式渲染模板的方法。

```py
# 流式渲染模板
def stream_template(template_name, **context):
    # 将 app 中的请求上下文内容更新至传入的上下文对象 context，
    # 这样确保请求上下文会传入即将被渲染的模板中
    app.update_template_context(context)
    # 获取 Jinja2 的模板对象
    template = app.jinja_env.get_template(template_name)
    # 获取流式渲染模板的生成器
    generator = template.stream(context)
    # 启用缓存，这样不会每一条都发送，而是缓存满了再发送
    generator.enable_buffering(5)

    return generator

```

这段代码的核心，就是通过”app.jinja_env”来访问 Jinja2 的 Environment 对象，这个我们在 Jinja2 系列中有介绍，然后调用 Environment 对象的”get_template()”方法来获得模板对象，再调用模板对象的”stream()”方法生成一个”StreamTemplate”的对象。这个对象实现了”__next__()”方法，可以作为一个生成器使用，如果你看了 Jinja2 的源码，你会发现模板对象的”stream()”方法的实现就是使用了 yield 表达式，所以原理同上例一样。另外，我们启用了缓存”enable_buffering()”来避免客户端发送过于频繁，其参数的默认值就是 5。

现在我们就可以在视图方法中，采用”stream_template()”，而不是以前介绍的”render_template()”来渲染模板了：

```py
@app.route('/stream.html')
def render_large_template():
    file = open('server.log')
    return Response(stream_template('stream-view.html',
                                    logs=file.readlines()))

```

上例的代码会将本地的”server.log”日志文件内容传入模板，并以流的方式渲染在页面上。

另外注意，在生成器中是无法访问请求上下文的。不过 Flask 从版本 0.9 开始提供了”stream_with_context()”方法，它允许生成器在运行期间获取请求上下文：

```py
from flask import request, stream_with_context

@app.route('/method')
def streamed_response():
    def generate():
        yield 'Request method is: '
        yield request.method
        yield '.'
    return Response(stream_with_context(generate()))

```

因为我们初始化 Response 对象时调用了”stream_with_context()”方法，所以才能在 yield 表达式中访问 request 对象。

### 文件上传

我们分下面 4 个步骤来实现文件上传功能：

1.  首先建立一个让用户上传文件的页面，我们将其放在模板”upload.html”中

```py
<!DOCTYPE html>
<title>Upload File</title>
<h1>Upload new File</h1>
<form action="" method="post" enctype="multipart/form-data">
  <p><input type="file" name="file">
     <input type="submit" value="Upload">
</form>

```

这里主要就是一个 enctype=”multipart/form-data”的 form 表单；一个类型为 file 的 input 框，即文件选择框；还有一个提交按钮。

*   定义一个文件合法性检查函数

```py
# 设置允许上传的文件类型
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg'])

# 检查文件类型是否合法
def allowed_file(filename):
    # 判断文件的扩展名是否在配置项 ALLOWED_EXTENSIONS 中
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

```

这样避免用户上传脚本文件如 js 或 php 文件，来干坏事！

*   文件提交后，在 POST 请求的视图函数中，通过 request.files 获取文件对象

这个 request.files 是一个字典，字典的键值就是之前模板中文件选择框的”name”属性的值，上例中是”file”；键值所对应的内容就是上传过来的文件对象。

*   检查文件对象的合法性后，通过文件对象的 save()方法将文件保存在本地

我们将第 3 和第 4 步都放在视图函数中，代码如下：

```py
import os
from flask import flask, render_template
from werkzeug import secure_filename

app = Flask(__name__)
# 设置请求内容的大小限制，即限制了上传文件的大小
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

# 设置上传文件存放的目录
UPLOAD_FOLDER = './uploads'

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # 获取上传过来的文件对象
        file = request.files['file']
        # 检查文件对象是否存在，且文件名合法
        if file and allowed_file(file.filename):
            # 去除文件名中不合法的内容
            filename = secure_filename(file.filename)
            # 将文件保存在本地 UPLOAD_FOLDER 目录下
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            return 'Upload Successfully'
        else:    # 文件不合法
            return 'Upload Failed'
    else:    # GET 方法
        return render_template('upload.html')

```

解释都写在注释里了，应该很容易看懂。重点要介绍的就是：

1.  Flask 的 MAX_CONTENT_LENGTH 配置项可以限制请求内容的大小

默认是没有限制，上例中我们设为 5M。

3.  必须调用”werkzeug.secure_filename()”来使文件名安全

比如用户上传的文件名为”../../../../home/username/.bashrc”，”secure_filename()”方法可以将其转为”home_username_.bashrc”。还是用来避免用户干坏事！

5.  Flask 处理文件上传的方式

如果上传的文件很小，那么会把它直接存在内存中。否则就会把它保存到一个临时目录下，通过”tempfile.gettempdir()”方法可以获取这个临时目录的位置。

另外 Flask 从 0.5 版本开始提供了一个简便的方法来让用户获取已上传的文件：

```py
from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

```

这个帮助方法”send_from_directory()”可以安全地将文件发送给客户端，它还可以接受一个参数”mimetype”来指定文件类型，和参数”as_attachment=True”来添加响应头”Content-Disposition: attachment”。

本篇中的示例代码可以在这里下载。

转载请注明出处: [思诚之道](http://www.bjhee.com/flask-ad5.html)