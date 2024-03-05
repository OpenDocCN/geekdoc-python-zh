# Django Rest 框架——简介

> 原文：<https://realpython.com/django-rest-framework-quick-start/>

让我们看看如何使用 [Django Rest Framework](http://www.django-rest-framework.org/) (DRF)为我们的 Django Talk 项目创建 RESTFul API，这是一个用于基于 Django 模型快速构建 RESTFul API 的应用程序。

换句话说，我们将使用 DRF 把一个非 RESTful 应用程序转换成 RESTful 应用程序。对于这个应用程序，我们将使用 DRF 版本 2.4.2。

本教程涵盖了这些主题:

1.  DRF 设置
2.  宁静的结构
3.  模型序列化程序
4.  DRF 网络可浏览 API

**免费奖励:** ，并获得 Python + REST API 原则的实际操作介绍以及可操作的示例。

如果你错过了这个系列教程的第一部分[和第二部分](https://realpython.com/django-and-ajax-form-submissions/)和第三部分[，一定要去看看。需要密码吗？从](https://realpython.com/django-and-ajax-form-submissions-more-practice/)[回购](https://github.com/realpython/django-form-fun)下载。要获得关于 Django Rest 框架的更深入的教程，请查看第三个 [Real Python](https://realpython.com/products/real-python-course/) 课程。

## DRF 设置

安装:

```py
$ pip install djangorestframework
$ pip freeze > requirements.txt
```

更新 *settings.py* :

```py
INSTALLED_APPS = (
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'talk',
    'rest_framework'
)
```

嘣！

[*Remove ads*](/account/join/)

## RESTful 结构

在 RESTful API 中，端点(URL)定义了 API 的结构以及最终用户如何使用 HTTP 方法从我们的应用程序访问数据:GET、POST、PUT、DELETE。端点应该围绕*集合*和*元素*进行逻辑组织，两者都是资源。在我们的例子中，我们有一个单独的资源，`posts`，所以我们将使用下面的 URL-`/posts/`和`/posts/<id>`，分别用于集合和元素。

|  | 得到 | 邮政 | 放 | 删除 |
| --- | --- | --- | --- | --- |
| `/posts/` | 显示所有帖子 | 添加新帖子 | 更新所有帖子 | 删除所有帖子 |
| `/posts/<id>` | 显示`<id>` | 不适用的 | 更新`<id>` | 删除`id` |

## DRF 快速启动

让我们启动并运行我们的新 API 吧！

### 模型串行器

DRF 的序列化器将模型实例转换成 Python 字典，然后可以用各种 API 适当的格式呈现——像 [JSON](https://realpython.com/python-json/) 或 XML。类似于 Django `ModelForm`类，DRF 为其序列化器提供了一种简洁的格式，即`ModelSerializer`类。使用起来很简单:只需告诉它您想要使用模型中的哪些字段:

```py
from rest_framework import serializers
from talk.models import Post

class PostSerializer(serializers.ModelSerializer):

    class Meta:
        model = Post
        fields = ('id', 'author', 'text', 'created', 'updated')
```

在“talk”目录中将其保存为*serializer . py*。

### 更新视图

我们需要重构当前的视图，以适应 RESTful 范式。注释掉当前视图并添加:

```py
from django.shortcuts import render
from django.http import HttpResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from talk.models import Post
from talk.serializers import PostSerializer
from talk.forms import PostForm

def home(request):
    tmpl_vars = {'form': PostForm()}
    return render(request, 'talk/index.html', tmpl_vars)

@api_view(['GET'])
def post_collection(request):
    if request.method == 'GET':
        posts = Post.objects.all()
        serializer = PostSerializer(posts, many=True)
        return Response(serializer.data)

@api_view(['GET'])
def post_element(request, pk):
    try:
        post = Post.objects.get(pk=pk)
    except Post.DoesNotExist:
        return HttpResponse(status=404)

    if request.method == 'GET':
        serializer = PostSerializer(post)
        return Response(serializer.data)
```

**这里发生了什么**:

1.  首先，`@api_view` decorator 检查适当的 HTTP 请求是否被传递到视图函数中。目前，我们只支持 GET 请求。
2.  然后，视图要么获取所有数据(如果是集合的话),要么只获取一个帖子(如果是元素的话)。
3.  最后，数据被序列化为 JSON 并返回。

> 请务必从官方[文档](http://www.django-rest-framework.org/api-guide/views#@api_view)中阅读更多关于`@api_view`的内容。

### 更新网址

让我们连接一些新的网址:

```py
# Talk urls
from django.conf.urls import patterns, url

urlpatterns = patterns(
    'talk.views',
    url(r'^$', 'home'),

    # api
    url(r'^api/v1/posts/$', 'post_collection'),
    url(r'^api/v1/posts/(?P<pk>[0-9]+)$', 'post_element')
)
```

### 测试

我们现在已经准备好[我们的第一次测试](https://realpython.com/python-testing/)！

1.  启动服务器，然后导航到:[http://127 . 0 . 0 . 1:8000/API/v1/posts/？format=json](http://127.0.0.1:8000/api/v1/posts/?format=json) 。
2.  现在让我们来看看[可浏览的 API](http://www.django-rest-framework.org/topics/browsable-api) 。导航到[http://127 . 0 . 0 . 1:8000/API/v1/posts/](http://127.0.0.1:8000/api/v1/posts/)

    因此，我们不需要做额外的工作，就可以自动获得这个漂亮的、人类可读的 API 输出。不错！这对 DRF 来说是一个巨大的胜利。

3.  元素怎么样？试试:[http://127 . 0 . 0 . 1:8000/API/v1/posts/1](http://127.0.0.1:8000/api/v1/posts/1)

在继续之前，您可能已经注意到作者字段是一个`id`而不是实际的`username`。我们将很快解决这个问题。现在，让我们连接新的 API，以便它可以与当前应用程序的模板一起工作。

[*Remove ads*](/account/join/)

## REST 重构

### 获取

在初始页面加载时，我们希望显示所有的文章。为此，添加以下 AJAX 请求:

```py
load_posts() // Load all posts on page load function  load_posts()  { $.ajax({ url  :  "api/v1/posts/",  // the endpoint type  :  "GET",  // http method // handle a successful response success  :  function(json)  { for  (var  i  =  0;  i  <  json.length;  i++)  { console.log(json[i]) $("#talk").prepend("<li id='post-"+json[i].id+"'><strong>"+json[i].text+"</strong> - <em> "+json[i].author+"</em> - <span> "+json[i].created+ "</span> - <a id='delete-post-"+json[i].id+"'>delete me</a></li>"); } }, // handle a non-successful response error  :  function(xhr,errmsg,err)  { $('#results').html("<div class='alert-box alert radius' data-alert>Oops! We have encountered an error: "+errmsg+ " <a href='#' class='close'>&times;</a></div>");  // add the error to the dom console.log(xhr.status  +  ": "  +  xhr.responseText);  // provide a bit more info about the error to the console } }); };
```

这些你以前都见过。注意我们是如何处理成功的:由于 API 发回了许多对象，我们需要遍历它们，将每个对象追加到 DOM 中。当我们连载帖子`id`时，我们也将`json[i].postpk`改为`json[i].id`。

测试一下。启动服务器，登录，然后查看帖子。

除了显示为`id`的`author`之外，请注意日期时间格式。这不是我们想要的，对吗？我们想要一个*可读的*日期时间格式。让我们更新一下…

### Datetime Format

我们可以使用一个名为 [MomentJS](http://momentjs.com/) 的强大的 [JavaScript](https://realpython.com/python-vs-javascript/) 库来轻松地按照我们想要的方式[格式化日期](https://realpython.com/python-time-module/)。

首先，我们需要将库导入到我们的*index.html*文件中:

```py
<!-- scripts -->
<script src="http://cdnjs.cloudflare.com/ajax/libs/moment.js/2.8.2/moment.min.js"></script>
<script src="static/scripts/main.js"></script>
```

然后更新 *main.js* 中的 for 循环:

```py
for  (var  i  =  0;  i  <  json.length;  i++)  { dateString  =  convert_to_readable_date(json[i].created) $("#talk").prepend("<li id='post-"+json[i].id+"'><strong>"+json[i].text+ "</strong> - <em> "+json[i].author+"</em> - <span> "+dateString+ "</span> - <a id='delete-post-"+json[i].id+"'>delete me</a></li>"); }
```

这里我们将日期字符串传递给一个名为`convert_to_readable_date()`的新函数，这个函数需要添加:

```py
// convert ugly date to human readable date function  convert_to_readable_date(date_time_string)  { var  newDate  =  moment(date_time_string).format('MM/DD/YYYY, h:mm:ss a') return  newDate }
```

就是这样。刷新浏览器。日期时间格式现在应该类似于这样- `08/22/2014, 6:48:29 pm`。请务必查看 [MomentJS](http://momentjs.com/) 文档，查看关于用 JavaScript 解析和格式化日期时间字符串的更多信息。

### 帖子

POST 请求以类似的方式处理。在使用序列化程序之前，让我们先通过更新视图来测试它。也许我们会很幸运，它会工作。

更新 *views.py* 中的`post_collection()`函数:

```py
@api_view(['GET', 'POST'])
def post_collection(request):
    if request.method == 'GET':
        posts = Post.objects.all()
        serializer = PostSerializer(posts, many=True)
        return Response(serializer.data)
    elif request.method == 'POST':
        data = {'text': request.DATA.get('the_post'), 'author': request.user.pk}
        serializer = PostSerializer(data=data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
```

还要添加以下导入内容:

```py
from rest_framework import status
```

**这里发生了什么**:

1.  `request.DATA`扩展 Django 的 HTTPRequest，从请求体返回内容。点击了解更多信息。
2.  如果反序列化过程有效，我们返回一个代码为 201(已创建)的响应。
3.  另一方面，如果反序列化过程失败，我们返回 400 响应。

更新`create_post()`功能中的端点

出发地:

```py
url  :  "create_post/",  // the endpoint
```

收件人:

```py
url  :  "api/v1/posts/",  // the endpoint
```

在浏览器中测试它。应该能行。不要忘记正确更新日期的处理，以及将`json.postpk`更改为`json.id`:

```py
success  :  function(json)  { $('#post-text').val('');  // remove the value from the input console.log(json);  // log the returned json to the console dateString  =  convert_to_readable_date(json.created) $("#talk").prepend("<li id='post-"+json.id+"'><strong>"+json.text+"</strong> - <em> "+ json.author+"</em> - <span> "+dateString+ "</span> - <a id='delete-post-"+json.id+"'>delete me</a></li>"); console.log("success");  // another sanity check },
```

[*Remove ads*](/account/join/)

### 作者格式

现在是停下来解决作者`id`对`username`问题的好时机。我们有几个选择:

1.  真正的 RESTFUL 并进行另一个调用来获取用户信息，这对性能没有好处。
2.  利用 [SlugRelatedField](http://www.django-rest-framework.org/api-guide/relations#slugrelatedfield) 关系。

让我们选择后者。更新序列化程序:

```py
from django.contrib.auth.models import User
from rest_framework import serializers
from talk.models import Post

class PostSerializer(serializers.ModelSerializer):
    author = serializers.SlugRelatedField(
        queryset=User.objects.all(), slug_field='username'
    )

    class Meta:
        model = Post
        fields = ('id', 'author', 'text', 'created', 'updated')
```

这里发生了什么事？

1.  `SlugRelatedField`允许我们将`author`字段的目标从`id`更改为`username`。
2.  此外，默认情况下，目标字段`username`既可读又可写，因此这种关系对于 get 和 POST 请求都是现成的。

更新视图中的`data`变量:

```py
data = {'text': request.DATA.get('the_post'), 'author': request.user}
```

再次测试。你现在应该看到作者的`username`。确保 GET 和 POST 请求都正常工作。

### 删除

在改变或添加任何东西之前，先测试一下。尝试删除链接。会发生什么？您应该会得到一个 404 错误。知道为什么会这样吗？或者去哪里找出问题所在？我们的 JavaScript 文件中的`delete_post`函数怎么样:

```py
url  :  "delete_post/",  // the endpoint
```

该 URL 不存在。在我们更新它之前，问问你自己——“我们应该针对集合还是单个元素？”。如果您不确定，请向上滚动查看 *RESTful 结构*表。除非我们想删除*所有的*帖子，那么我们需要点击元素端点:

```py
url  :  "api/v1/posts/"+post_primary_key,  // the endpoint
```

再次测试。现在发生了什么？您应该会看到一个 405 错误- `405: {"detail": "Method 'DELETE' not allowed."}` -因为视图没有设置为处理删除请求。

```py
@api_view(['GET', 'DELETE'])
def post_element(request, pk):
    try:
        post = Post.objects.get(pk=pk)
    except Post.DoesNotExist:
        return HttpResponse(status=404)

    if request.method == 'GET':
        serializer = PostSerializer(post)
        return Response(serializer.data)

    elif request.method == 'DELETE':
        post.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
```

添加了 DELETE HTTP 谓词后，我们可以通过用`delete()`方法删除 post 并返回 204 响应来处理请求。有用吗？只有一个办法可以知道。这一次当你测试时，确保(a)这篇文章确实被删除并从 DOM 中移除，以及(b)204 状态码被返回(你可以在 *Chrome 开发者工具*的*网络*标签中确认这一点)。

## 结论和后续步骤

**免费奖励:** ，并获得 Python + REST API 原则的实际操作介绍以及可操作的示例。

暂时就这些了。需要额外的挑战吗？添加使用 PUT 请求更新帖子的功能。

实际的 REST 部分很简单:您只需要更新`post_element()`函数来处理 PUT 请求。

客户端有点困难，因为您需要更新实际的 HTML 来显示一个输入框，供用户输入新值，您需要在 JavaScript 文件中获取这个输入框，这样您就可以将它与 put 请求一起发送。

您是否允许任何用户更新任何帖子，而不管该帖子最初是否是她/他发布的？

如果是，你打算更新作者姓名吗？也许在数据库中添加一个`edited_by`字段？然后在 DOM 上显示编辑过的注释。如果用户只能更新他们自己的帖子，您需要确保在视图中正确处理这一点，然后在用户试图编辑他/她最初没有发布的帖子时显示一条错误消息。

或者您可以删除某个用户不能编辑的帖子的编辑链接(也可以删除链接)。你可以把它变成一个权限问题，只让*某些*用户，如版主或管理员，编辑所有帖子，而其余用户只能更新他们自己的帖子。

这么多问题。

如果你决定尝试这种方法，那就选择最容易实现的——简单地允许任何用户更新任何帖子，并且只更新数据库中的`text`。然后测试。然后添加另一个迭代。然后测试等等。做好笔记，给 info@realpython.com 发邮件，这样我们就可以补充一篇博文了！

无论如何，下一次你将会看到我们在添加 **Angular** 时分解当前的 JavaScript 代码！到时候见。***