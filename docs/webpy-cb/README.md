# Web.py Cookbook 简体中文版

> 来源：[Web.py Cookbook 简体中文版](http://webpy.org/cookbook/index.zh-cn)

欢迎来到 web.py 0.3 的 Cookbook。提醒您注意：某些特性在之前的版本中并不可用。当前开发版本是 0.3。

## 格式

1.  在编排内容时，请尽量使用 cookbook 格式...如：

    ### 问题：如何访问数据库中的数据？

    ### 解法：使用如下代码...

2.  请注意，网址中不必含有"web"。如"/cookbook/select"，而非"/cookbook/web.select"。

3.  该手册适用于 0.3 版本，所以您在添加代码时，请确认代码能在新版本中工作。

* * *

## 基本应用:

*   Hello World
*   提供静态文件访问
*   理解 URL 控制
*   跳转与重定向
*   使用子应用
*   提供 XML 访问
*   从 post 读取原始数据

## 高级应用

*   用 web.ctx 获得客户端信息
*   应用处理器，添加钩子和卸载钩子
*   如何使用 web.background
*   自定义 NotFound 信息
*   如何流传输大文件
*   对自带的 webserver 日志进行操作
*   用 cherrypy 提供 SSL 支持
*   实时语言切换

## Sessions and user state 会话和用户状态:

*   如何使用 Session
*   如何在调试模式下使用 Session
*   在 template 中使用 session
*   如何操作 Cookie
*   用户认证
*   一个在 postgreSQL 数据库环境下的用户认证的例子
*   如何在子应用中操作 Session

## Utils 实用工具:

*   如何发送邮件
*   如何利用 Gmail 发送邮件
*   使用 soaplib 实现 webservice

## Templates 模板

*   Templetor: web.py 模板系统
*   使用站点布局模板
*   交替式风格 (未译)
*   导入函数到模板中 (未译)
*   模板文件中的 i18n 支持
*   在 web.py 中使用 Mako 模板引擎
*   在 web.py 中使用 Cheetah 模板引擎
*   在 web.py 中使用 Jinja2 模板引擎
*   如何在谷歌应用程序引擎使用模板

## Testing 测试:

*   Testing with Paste and Nose (未译)
*   RESTful doctesting using an application's request method (未译)

## User input 用户输入:

*   文件上传
*   保存上传的文件
*   上传文件大小限定
*   通过 web.input 接受用户输入
*   怎样使用表单
*   显示个别表单字段

## Database 数据库

*   使用多数据库
*   Select: 查询数据
*   Update: 更新数据
*   Delete: 删除数据
*   Insert: 新增数据
*   Query: 高级数据库查询
*   怎样使用数据库事务
*   使用 sqlalchemy
*   整合 SQLite UDF (用户定义函数) 到 webpy 数据库层
*   使用字典动态构造 where 子句

## Deployment 部署:

*   通过 Fastcgi 和 lighttpd 部署
*   通过 Webpy 和 Nginx with FastCGI 搭建 Web.py
*   CGI deployment through Apache (未译)
*   mod_python deployment through Apache (requested)
*   通过 Apache 和 mod_wsgi 部署
*   mod_wsgi deployment through Nginx (未译)
*   Fastcgi deployment through Nginx (未译)

## Subdomains 子域名:

*   Subdomains and how to access the username (requested)