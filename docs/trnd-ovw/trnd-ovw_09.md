# 注意事项和社区支持

## 注意事项和社区支持

因为 FriendFeed 以及其他 Tornado 的主要用户在使用时都是基于 nginx 或者 Apache 代理之后的。所以现在 Tornado 的 HTTP 服务部分并不完整，它无法处理多行的 header 信息，同时对于一 些非标准的输入也无能为力。

你可以在 [Tornado 开发者邮件列表](http://groups.google.com/group/python-tornado) 中讨论和提交 bug。