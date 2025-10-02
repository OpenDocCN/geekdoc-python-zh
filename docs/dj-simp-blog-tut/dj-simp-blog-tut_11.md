# 搜索和 ReadMore

# 搜索功能

搜索功能的实现设计:

*   前段界面输入搜索关键字, 传送到对应 view 中
*   在对应的 view 中进行数据库关键字搜索
*   这里搜索可以只对文章名搜索或者全文搜索

首先在 my_blog/templates 下添加所有输入框

```py
<div class="sidebar pure-u-1 pure-u-md-1-4">
        <div class="header">
            <h1 class="brand-title"><a href="{% url "home" %}">Andrew Liu Blog</a></h1>
            <h2 class="brand-tagline">雪忆 - Snow Memory</h2>
            <nav class="nav">
                <ul class="nav-list">
                    <li class="nav-item">
                        <a class="button-success pure-button" href="/">主页</a>
                    </li>
                    <li class="nav-item">
                        <a class="button-success pure-button" href="{% url "archives" %}">归档</a>
                    </li>
                    <li class="nav-item">
                        <a class="pure-button" href="https://github.com/Andrew-liu/my_blog_tutorial">Github</a>
                    </li>
                    <li class="nav-item">
                        <a class="button-error pure-button" href="http://weibo.com/dinosaurliu">Weibo</a>
                    </li>
                    <li class="nav-item">
                        <a class="button-success pure-button" href="/">专题</a>
                    </li>
                    <li>
                    <form class="pure-form" action="/search/" method="get">
                    <input class="pure-input-3-3" type="text" name="s" placeholder="search">
                    </form>
                    </li>
                    <li class="nav-item">
                        <a class="button-success pure-button" href="{% url "about_me" %}">About Me</a>
                    </li>
                </ul>
            </nav>
        </div>
    </div> 
```

在 my_blog/article/views.py 中添加查询逻辑

```py
def blog_search(request):
    if 's' in request.GET:
        s = request.GET['s']
        if not s:
            return render(request,'home.html')
        else:
            post_list = Article.objects.filter(title__icontains = s)
            if len(post_list) == 0 :
                return render(request,'archives.html', {'post_list' : post_list,
                                                    'error' : True})
            else :
                return render(request,'archives.html', {'post_list' : post_list,
                                                    'error' : False})
    return redirect('/') 
```

这里为了简单起见, 直接对`archives.html`进行修改, 使其符合查询逻辑

```py
{% extends "base.html" %}

{% block content %}
<div class="posts">
    {% if error %}
        <h2 class="post-title">没有相关文章题目</a></h2>
    {% else %}
    {% for post in post_list %}
        <section class="post">
            <header class="post-header">
                <h2 class="post-title"><a href="{% url "detail" id=post.id %}">{{ post.title }}</a></h2>

                    <p class="post-meta">
                        Time:  <a class="post-author" href="#">{{ post.date_time |date:"Y /m /d"}}</a> <a class="post-category post-category-js" href="{% url "search_tag" tag=post.category %}">{{ post.category }}</a>
                    </p>
            </header>
        </section>
    {% endfor %}
    {% endif %}
</div><!-- /.blog-post -->
{% endblock %} 
```

添加了 if 判断逻辑, 然后还需要修改`views 中的 archives`

```py
def archives(request) :
    try:
        post_list = Article.objects.all()
    except Article.DoesNotExist :
        raise Http404
    return render(request, 'archives.html', {'post_list' : post_list, 
                                            'error' : False}) 
```

最后添加`my_blog/my_blog/urls.py`设置 url

```py
urlpatterns = patterns('',
    # Examples:
    # url(r'^$', 'my_blog.views.home', name='home'),
    # url(r'^blog/', include('blog.urls')),

    url(r'^admin/', include(admin.site.urls)),
    url(r'^$', 'article.views.home', name = 'home'),
    url(r'^(?P<id>\d+)/$', 'article.views.detail', name='detail'),
    url(r'^archives/$', 'article.views.archives', name = 'archives'),
    url(r'^aboutme/$', 'article.views.about_me', name = 'about_me'),
    url(r'^tag(?P<tag>\w+)/$', 'article.views.search_tag', name = 'search_tag'),
    url(r'^search/$','article.views.blog_search', name = 'search'),
) 
```

# ReadMore 功能

对于 ReadMore 的前段按钮界面设置早已经添加过了, 所以这里只需要进行简单的设置就好了

通过使用 Django 中内建的 filter 就可以速度实现

```py
{{ value|truncatewords:2 }} #这里 2 表示要显示的单词数, 以后的会被截断, 不在显示 
```

这里只需要修改 my_blog/templates/home.html 界面中的变量的过滤器

```py
#将正文截断设置为 10
 {{ post.content|custom_markdown|truncatewords_html:100 }} 
```

在浏览器中输入[`127.0.0.1:8000/`](http://127.0.0.1:8000/)可以看到效率(最好把博文设置的长一些)