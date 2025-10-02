# Flask 大型教程

翻译者注：本系列的原文名为：[The Flask Mega-Tutorial](http://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world) [http://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world] ，作者是 **Miguel Grinberg** 。

本系列是作者平时使用 Flask 微框架编写应用的经验之谈，这里是这一系列中所有已经发布的文章的索引。

*   Hello World
    *   作者背景
    *   应用程序简介
    *   要求
    *   安装 Flask
    *   在 Flask 中的 “Hello, World”
    *   下一步？
*   模板
    *   回顾
    *   为什么我们需要模板
    *   模板从天而降
    *   模板中控制语句
    *   模板中的循环语句
    *   模板继承
    *   结束语
*   web 表单
    *   回顾
    *   配置
    *   用户登录表单
    *   表单模板
    *   表单视图
    *   接收表单数据
    *   加强字段验证
    *   处理 OpenIDs
    *   结束语
*   数据库
    *   回顾
    *   从命令行中运行 Python 脚本
    *   Flask 中的数据库
    *   迁移
    *   配置
    *   数据库模型
    *   创建数据库
    *   第一次迁移
    *   数据库升级和回退
    *   数据库关系
    *   编程时间
    *   结束语
*   用户登录
    *   回顾
    *   配置
    *   重构用户模型
    *   user_loader 回调
    *   登录视图函数
    *   Flask-OpenID 登录回调
    *   全局变量 *g.user*
    *   首页视图
    *   登出
    *   结束语
*   用户信息页和头像
    *   回顾
    *   用户信息页
    *   头像
    *   在子模板中重用
    *   更多有趣的信息
    *   编辑用户信息
    *   结束语
*   单元测试
    *   回顾
    *   发现 bug
    *   Flask 调试
    *   定制 HTTP 错误处理器
    *   通过电子邮件发送错误
    *   记录到文件
    *   修复 bug
    *   单元测试框架
    *   结束语
*   关注者，联系人和好友
    *   回顾
    *   ‘关注者’ 特色的设计
    *   数据库关系
    *   表示关注者和被关注者
    *   数据模型
    *   添加和移除 ‘关注者’
    *   测试
    *   数据库查询
    *   可能的改进
    *   收尾
    *   结束语
*   分页
    *   回顾
    *   提交博客文章
    *   显示 blog
    *   分页
    *   页面导航
    *   实现 Post 子模板
    *   用户信息页
    *   结束语
*   全文搜索
    *   回顾
    *   全文搜索引擎的简介
    *   Python 3 兼容性
    *   配置
    *   模型修改
    *   搜索
    *   整合全文搜索到应用程序
    *   搜索结果页
    *   结束语
*   邮件支持
    *   回顾
    *   安装 Flask-Mail
    *   配置
    *   让我们发送邮件！
    *   简单的邮件框架
    *   关注提醒
    *   这就足够了吗？
    *   在 Python 中异步调用
    *   结束语
*   换装
    *   简介
    *   我们该怎么做？
    *   Bootstrap 简介
    *   用 Bootstrap 装点 *microblog*
    *   结束语
*   日期和时间
    *   善意提醒
    *   时间戳的问题
    *   用户特定的时间戳
    *   介绍 moment.js
    *   整合 moment.js
    *   结束语
*   国际化和本地化
    *   配置
    *   标记翻译文本
    *   提取文本翻译
    *   生成一个语言目录
    *   更新翻译
    *   翻译 *moment.js*
    *   惰性求值
    *   快捷方式
    *   结束语
*   Ajax
    *   客户端 VS 服务器端
    *   翻译用户生成内容
    *   确定 blog 语言
    *   显示 “翻译” 链接
    *   翻译服务
    *   使用 Microsoft Translator 服务
    *   让我们翻译一些文本
    *   服务器上的 Ajax
    *   客户端上的 Ajax
    *   结束语
*   调试，测试以及优化
    *   Bug
    *   现场调试问题
    *   使用 Python 调试器
    *   回归测试
    *   修复
    *   测试覆盖率
    *   性能调优
    *   数据库性能
    *   结束语

© Copyright Translate by by D.D 2013\. Created using [Sphinx](http://sphinx.pocoo.org/) 1.2\.

[京 ICP 备**********号](http://www.miitbeian.gov.cn/)

* * *