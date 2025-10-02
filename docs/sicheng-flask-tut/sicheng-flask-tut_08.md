# Flask 进阶系列(三)–Jinja2 模板引擎

其实我们在入门系列第三篇中已经介绍了模板，包括如何渲染模板，表达式和控制语句，模板继承，还有 HTML 转义。我们也知道了 Flask 模板是基于 Jinja2 实现的。其实 Jinja2 的模板功能远不止这些，想了想，还是决定在进阶系列中，更深入地介绍 Jinja2 模板引擎。

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

Jinja2 模板引擎同 Flask 一样，都是由一个叫[Pocoo](http://www.pocoo.org/)的组织维护着的，该组织另外几个有名的项目是[Werkzeug](http://werkzeug.pocoo.org/)和[Sphinx](http://www.sphinx-doc.org/en/stable/)。Werkzeug 是 Python 的 WSGI 规范的函数库，也是 Flask 的基础库之一。如果朋友们看过 Flask 源码的话，会发现其实 Flask 本身代码没多少，正如其所标榜的，就是一个非常轻量级的 Web 框架。Flask 很多实用的功能都是依赖 Werkzeug 和 Jinja2 实现的。Sphinx 是 Python 的文档生成工具，用其写文档语法很类似于 Markdown，Flask 官网的文档就是由 Sphinx 生成的。

讲了那么多不相关的，我们还是回到 Jinja2 上。从 Flask 源码上看，其只指明对”.html”, “.htm”, “.xml”, “.xhtml”这四种类型的文件开启 HTML 格式自动转义。所以，我们定义模板文件时最好选这些后缀名，个人建议就使用”.html”文件。另外，Flask 只选择加载了 2 个 Jinja 扩展，”jinja2.ext.autoescape”和”jinja2.ext.with_”，其他扩展功能则无法使用，我们回头会介绍如何对 Flask 应用添加 Jinja2 的扩展。你可以通过”app.jinja_env”（这里的 app 就是你的 Flask 应用对象）来访问 Jinja2 的对象。不过我们不建议这么做，Flask 开放了下面几个方法和装饰器来让应用开发者扩充 Jinja2 的功能：

| 函数 | 装饰器 | 作用 |
| add_template_filter | template_filter | 自定义过滤器 |
| add_template_test | template_test | 自定义测试器 |
| add_template_global | template_global | 自定义全局函数 |

本来计划在一篇文章里介绍 Jinja2 模板引擎在 Flask 里的使用，结果发现内容太多了，还是决定写一个系列（系列中套系列，汗…），读者看起来也轻松点。这里就列个大纲：

*   Flask 中 Jinja2 模板引擎详解(一)–控制语句和表达式
*   Flask 中 Jinja2 模板引擎详解(二)–上下文环境
*   Flask 中 Jinja2 模板引擎详解(三)–过滤器
*   Flask 中 Jinja2 模板引擎详解(四)–测试器
*   Flask 中 Jinja2 模板引擎详解(五)–全局函数
*   Flask 中 Jinja2 模板引擎详解(六)–块和宏
*   Flask 中 Jinja2 模板引擎详解(七)–本地化
*   Flask 中 Jinja2 模板引擎详解(八)–自定义扩展

另外，毕竟再怎么详细介绍，也无法比官网更全，所以还是那句老话，想再深入了解 Jinja2 的朋友们，可以参阅[官方文档](http://jinja.pocoo.org/)和[源代码](https://github.com/mitsuhiko/jinja2)。

转载请注明出处: [思诚之道](http://www.bjhee.com/flask-ad3.html)