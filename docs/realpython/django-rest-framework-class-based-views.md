# Django Rest 框架——基于类的视图

> 原文：<https://realpython.com/django-rest-framework-class-based-views/>

在这篇文章中，让我们继续开发 Django Talk 项目和 Django Rest 框架，同时实现基于 T2 类的视图，并做一些简单的重构来保持代码的干爽。**本质上，我们正在将现有的基于函数视图的 RESTful API 迁移到基于类的视图。**

> 这是本系列教程的第四部分。
> 
> **需要迎头赶上？**看看第一部分[第一部分](https://realpython.com/django-and-ajax-form-submissions/)和第二部分[第五部分](https://realpython.com/django-and-ajax-form-submissions-more-practice/)，我们将讨论 Django 和 AJAX，还有第三部分[第七部分，我们将介绍 Django Rest 框架。](https://realpython.com/django-rest-framework-quick-start/)
> 
> **需要代码吗？**从[回购](https://github.com/realpython/django-form-fun)下载。

要获得关于 Django Rest 框架的更深入的教程，请务必查看第三个 [Real Python](https://realpython.com) 课程。

## 重构

在进入基于类的视图之前，让我们对当前代码做一个[快速重构](https://realpython.com/python-refactoring/)。在`def post_element()`视图中，进行以下更新:

```py
post = get_object_or_404(Post, id=pk)

# try:
#     post = Post.objects.get(pk=pk)
# except Post.DoesNotExist:
#     return HttpResponse(status=404)
```

所以这里我们使用 [get_object_or_404](https://docs.djangoproject.com/en/1.6/topics/http/shortcuts/#get-object-or-404) 快捷方式来引发 404 错误而不是异常。

请确保也更新导入:

```py
from django.shortcuts import render, get_object_or_404
```

测试一下。尝试查看一个不存在的元素——即[http://localhost:8000/API/v1/posts/202？format=json](http://localhost:8000/api/v1/posts/202?format=json) 。您应该会在浏览器中看到以下响应:

```py
{ "detail":  "Not found" }
```

如果你在 *Chrome 开发者工具*中打开*网络*标签，你应该会看到一个 404 状态码。很酷，对吧？不幸的是，我们不会使用它太久，因为是时候告别我们当前的基于函数的视图并添加基于类的视图了。

[*Remove ads*](/account/join/)

## 基于类的视图

虽然函数很容易使用，但使用基于类的视图来重用功能通常是有益的，尤其是对于具有许多端点的大型 API。

注释掉基于函数的视图的代码。

### 收藏

为集合的基于类的视图添加以下代码:

```py
class PostCollection(mixins.ListModelMixin,
                     mixins.CreateModelMixin,
                     generics.GenericAPIView):

    queryset = Post.objects.all()
    serializer_class = PostSerializer

    def get(self, request, *args, **kwargs):
        return self.list(request, *args, **kwargs)

    def post(self, request, *args, **kwargs):
        return self.create(request, *args, **kwargs)
```

欢迎来到 [mixins](https://docs.djangoproject.com/en/1.6/ref/class-based-views/mixins/) 的力量！

1.  `ListModelMixin`提供了`list()`函数，用于将集合序列化为 JSON，然后将其作为 GET 请求的响应返回。
2.  同时，`CreateModelMixin`提供了用于创建新对象的`create()`函数，以响应 POST 请求。
3.  最后，`GenericAPIView` mixin 提供了 RESTful API 所需的“核心”功能。

请参考 DRF 官方文档了解更多关于这些混合的信息。

### 成员

现在为成员添加代码:

```py
class PostMember(mixins.RetrieveModelMixin,
                   mixins.DestroyModelMixin,
                   generics.GenericAPIView):

    queryset = Post.objects.all()
    serializer_class = PostSerializer

    def get(self, request, *args, **kwargs):
        return self.retrieve(request, *args, **kwargs)

    def delete(self, request, *args, **kwargs):
        return self.destroy(request, *args, **kwargs)
```

这里，我们简单地使用`GenericAPIView`作为“核心功能”,其余的 mixins 提供处理 GET 和 DELETE 请求所需的功能。

### URLs

最后，让我们更新 *urls.py* 来说明基于类的视图:

```py
from django.conf.urls import patterns, url
from talk import views

urlpatterns = patterns(
    'talk.views',
    url(r'^$', 'home'),

    # api
    url(r'^api/v1/posts/$', views.PostCollection.as_view()),
    url(r'^api/v1/posts/(?P<pk>[0-9]+)/$', views.PostMember.as_view())
)
```

请注意`as_view()`方法，它提供了一点魔力来将类作为视图函数处理。

测试一下。启动开发服务器，然后:

1.  确保所有帖子都已加载
2.  添加新帖子
3.  删除一个帖子

发生了什么事？尝试添加新帖子时，您应该会看到以下错误:

```py
400:  { "text":  ["This field is required."], "author":  ["This field is required."] }
```

幸运的是，这很容易解决。

[*Remove ads*](/account/join/)

### 重构 AJAX

我们需要改变发送 POST 请求数据的方式。首先，将`the_post`键更新为`text`:

```py
data  :  {  text  :  $('#post-text').val()  },  // data sent with the post request
```

如果您现在测试它，错误应该只是表明我们缺少了`author`字段。我们可以通过多种方式获取登录的用户，但是最简单的方式是直接从 DOM 获取。

> 值得注意的是，我们可以覆盖视图中的默认功能，从请求对象中获取用户名。然而，最好按照预期使用基于 DRF 类的视图:在 JSON 请求中传递所有适当的参数——例如，`text`和`author`——然后使用序列化程序保存它们。

在“模板/对话”目录下打开*index.html*。在文件的顶部，您会看到我们直接从`request`对象访问用户名:

```py
<h2>Hi, {{request.user.username}}</h2>
```

让我们隔离实际的用户名，以便更容易用 jQuery 获取:

```py
<h2>Hi, <span id="user">{{request.user.username}}</span></h2>
```

现在再次更新`data`:

```py
data  :  {  text  :  $('#post-text').val(),  author:  $('#user').text()}
```

测试一下；一切都会好的。

## 基于通用的视图

想要事半功倍？看看这个。注释掉我们刚刚添加的代码，然后像这样更新视图:

```py
from talk.models import Post
from talk.forms import PostForm
from talk.serializers import PostSerializer
from rest_framework import generics
from django.shortcuts import render

def home(request):
    tmpl_vars = {'form': PostForm()}
    return render(request, 'talk/index.html', tmpl_vars)

#########################
### class based views ###
#########################

class PostCollection(generics.ListCreateAPIView):
    queryset = Post.objects.all()
    serializer_class = PostSerializer

class PostMember(generics.RetrieveUpdateDestroyAPIView):
    queryset = Post.objects.all()
    serializer_class = PostSerializer
```

因此，我们不仅可以像以前一样处理所有相同的请求，还可以处理更新每个成员的 PUT 请求。更多。给**很多**少。

在继续之前，跳回基于函数的视图，将代码与基于类的视图进行比较。哪个更容易阅读？其实哪个更好理解？如有必要，添加注释，以帮助您更好地理解基于类的视图背后发生的事情。

你不仅牺牲了基于类的视图的可读性，而且[测试](https://realpython.com/python-testing/)也更加困难。然而，我们现在利用了[继承](https://realpython.com/inheritance-composition-python/)，对于有许多相似视图的大型项目，基于类的视图是完美的，因为你不必一遍又一遍地写相同的代码。**在跳到基于阶级的观点之前，一定要权衡利弊。**

在继续之前，一定要测试端点。

1.  确保所有帖子都已加载
2.  添加新帖子
3.  删除一个帖子

对 PUT 请求感到好奇？通过 HTML 表单从可浏览的 API(即[http://localhost:8000/API/v1/posts/1/](http://localhost:8000/api/v1/posts/1/))中测试它。想要完全删除 PUT 方法处理程序吗？像这样更新代码:

```py
class PostMember(generics.RetrieveDestroyAPIView):
    queryset = Post.objects.all()
    serializer_class = PostSerializer
```

现在测试一下。有问题吗？查看[文档](http://www.django-rest-framework.org/api-guide/generic-views#retrievedestroyapiview)。

[*Remove ads*](/account/join/)

## 结论

暂时就这些了。在以后的文章中，我们可能会跳回到 Django Rest 框架来看看分页、权限和基本验证，但在此之前，我们将在前端添加 AngularJS 来消费数据。

干杯！***