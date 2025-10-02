# User input 用户输入

# File Upload Recipe

## 问题

如果你不是很了解表单上传或者 CGI 的话, 你会觉得文件上传有点奇特.

## 解决方法

```py
import web

urls = ('/upload', 'Upload')

class Upload:
    def GET(self):
        return """<html><head></head><body>
<form method="POST" enctype="multipart/form-data" action="">
<input type="file" name="myfile" />
<br/>
<input type="submit" />
</form>
</body></html>"""

    def POST(self):
        x = web.input(myfile={})
        web.debug(x['myfile'].filename) # 这里是文件名
        web.debug(x['myfile'].value) # 这里是文件内容
        web.debug(x['myfile'].file.read()) # 或者使用一个文件对象
        raise web.seeother('/upload')

if __name__ == "__main__":
   app = web.application(urls, globals()) 
   app.run() 
```

## 注意

需要注意以下内容:

*   表单需要一个 enctype="multipart/form-data"的属性, 否则不会正常工作.
*   在 webpy 的代码里, 如果你需要默认值的话, myfile 就需要默认值了(myfile={}), 文件会以字符串的形式传输 -- 这确实可以工作, 但是你会丢失文件的名称

# 保存上传的文件

## 问题

上传文件，并将其保存到预先设定的某个目录下。

## 方法

```py
import web

urls = ('/upload', 'Upload')

class Upload:
    def GET(self):
        web.header("Content-Type","text/html; charset=utf-8")
        return """<html><head></head><body>
<form method="POST" enctype="multipart/form-data" action="">
<input type="file" name="myfile" />
<br/>
<input type="submit" />
</form>
</body></html>"""

    def POST(self):
        x = web.input(myfile={})
        filedir = '/path/where/you/want/to/save' # change this to the directory you want to store the file in.
        if 'myfile' in x: # to check if the file-object is created
            filepath=x.myfile.filename.replace('\\','/') # replaces the windows-style slashes with linux ones.
            filename=filepath.split('/')[-1] # splits the and chooses the last part (the filename with extension)
            fout = open(filedir +'/'+ filename,'w') # creates the file where the uploaded file should be stored
            fout.write(x.myfile.file.read()) # writes the uploaded file to the newly created file.
            fout.close() # closes the file, upload complete.
        raise web.seeother('/upload')

if __name__ == "__main__":
   app = web.application(urls, globals()) 
   app.run() 
```

## Hang ups

同时还需要注意如下几点:

*   转到 fileupload。
*   千万不要让用户把文件上传到那些不经过文件后缀和类型检查而执行文件的文件夹下。
*   事实上，一定要以"mb"模式打开文件（在 windows 下）， 也就是二进制可写模式, 否则图片将无法上传。

# 上传文件大小限定

## 问题

如何限定上传文件的大小？

## Solution

web.py 使用`cgi` 模块来解析用户的输入， 而 `cgi` 模块对最大输入大小有限制。

下面的代码限制了最大数据输入为 10MB.

```py
import cgi

# Maximum input we will accept when REQUEST_METHOD is POST
# 0 ==> unlimited input
cgi.maxlen = 10 * 1024 * 1024 # 10MB 
```

请注意这是对 POST 方法提交数据大小的限制，而不是上传文件大小。当然如果表单中没有其他输入数据，上传文件完全可以达到限制的大小。

`cgi` 模块将会抛出 `ValueError`异常，如果数据输入的大小超过了 `cgi.maxlen`。我们可以捕捉该异常而避免显示不友好的错误信息。

```py
class upload:
    def POST(self):
        try:
            i = web.input(file={})
        except ValueError:
            return "File too large" 
```

# web.input

## web.input

### 问题

如何从 form 或是 url 参数接受用户数据.

### 解决方法

web.input()方法返回一个包含从 url(GET 方法)或 http header(POST 方法,即表单 POST)获取的变量的 web.storage 对象(类似字典).举个例子,如果你访问页面[`example.com/test?id=10,在 Python 后台你想取得`](http://example.com/test?id=10,在 Python 后台你想取得) id=10 ,那么通过 web.input()那就是小菜一碟:

```py
class SomePage:
    def GET(self):
        user_data = web.input()
        return "<h1>" + user_data.id + "</h1>" 
```

有时你想指定一个默认变量,而不想使用 None.参考下面的代码:

```py
class SomePage:
    def GET(self):
        user_data = web.input(id="no data")
        return "<h1>" + user_data.id + "</h1>" 
```

注意,web.input()取得的值都会被当作 string 类型,即使你传递的是一些数字.

如果你想传递一个多值变量,比如像这样:

<select multiple="" size="3"><option>foo</option><option>bar</option><option>baz</option></select>

你需要让 web.input 知道这是一个多值变量,否则会变成一串而不是一个变量 .传递一个 list 给 web.input 作为默认值,就会正常工作.举个例子, 访问 [`example.com?id=10&id=20`](http://example.com?id=10&id=20):

```py
class SomePage:
    def GET(self):
        user_data = web.input(id=[])
        return "<h1>" + ",".join(user_data.id) + "</h1>" 
```

译者补充: 多值变量这儿,在 WEB 上除了上面所说的 multiple select 和 query strings 外,用得最多的就是复选框(checkbox)了,另外还有多文件上传时的<input type="file" ...>.

# 怎样使用表单 forms

## 问题：

怎样使用表单 forms

## 解决：

'web.form'模块提供支持创建，校验和显示表单。该模块包含一个'Form'类和各种输入框类如'Textbox'，'Password'等等。

当'form.validates()'调用时，可以针对每个输入检测的哪个是有效的，并取得校验理由列表。

'Form'类同样可以使用完整输入附加的关键字参数'validators'来校验表单。

这里是一个新用户注册的表单的示例：

```py
import web
from web import form

render = web.template.render('templates') # your templates

vpass = form.regexp(r".{3,20}$", 'must be between 3 and 20 characters')
vemail = form.regexp(r".*@.*", "must be a valid email address")

register_form = form.Form(
    form.Textbox("username", description="Username"),
    form.Textbox("email", vemail, description="E-Mail"),
    form.Password("password", vpass, description="Password"),
    form.Password("password2", description="Repeat password"),
    form.Button("submit", type="submit", description="Register"),
    validators = [
        form.Validator("Passwords did't match", lambda i: i.password == i.password2)]

)

class register:
    def GET(self):
        # do $:f.render() in the template
        f = register_form()
        return render.register(f)

    def POST(self):
        f = register_form()
        if not f.validates():
            return render.register(f)
        else:
            # do whatever is required for registration 
```

然后注册的模板应该像是这样：

```py
$def with(form)

<h1>Register</h1>
<form method="POST">
    $:form.render()
</form> 
```

# 个别显示表单字段

### 问题：

怎样在模板中个别显示表单字段？

### 解决：

你可以使用'render()'方法在你的模板中显示部分的表单字段。

假设你想创建一个名字/姓氏表单。很简单，只有两个字段，不需要验证，只是为了测试目的。

```py
from web import form
simple_form = form.Form(
    form.Textbox('name', description='Name'),
    form.Textbox('surname', description='Surname'),
) 
```

通常你可以使用`simple_form.render（）`或`simple_form.render_css（）`。 但如你果你想一个一个的显示表单的字段，或者你怎样才能对模板中的表单显示拥有更多的控制权限？如果是这样，你可以对你的个别字段使用`render()`方法。

我们定义了两个字段名称为`name`和`surname`。这些名称将自动成为`simple_form`对象的属性。

```py
>>> simple_form.name.render()
'<input type="text" name="name" id="name" />'
>>> simple_form.surname.render()
'<input type="text" name="surname" id="surname" />' 
```

你同样可以通过类似的方法显示个别的描述：

```py
>>> simple_form.surname.description
'Surname' 
```

如果你有一个小模板片段（局部模板），你想统一的使用你所定义的所有表单字段？你可以使用表单对象的`inputs`属性迭代每个字段。下面是一个示例：

```py
>>> for input in simple_form.inputs:
...     print input.description
...     print input.render()
... 
Name
<input type="text" name="name" id="name" />
Surname
<input type="text" name="surname" id="surname" /> 
```