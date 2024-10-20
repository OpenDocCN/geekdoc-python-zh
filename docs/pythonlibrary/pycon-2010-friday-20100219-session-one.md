# PyCon 2010:星期五(2010/02/19) -第一场会议

> 原文：<https://www.blog.pythonlibrary.org/2010/02/19/pycon-2010-friday-20100219-session-one/>

我上午参加了三个会议:建立 Leafy Chat，一个简短的 Pinax 教程和 Import This，that 和其他东西:自定义进口商。如果你感兴趣，你可以继续读下去，看看我是怎么想的。

Alex Gaynor 使用 AJAX 和 Comet 在“实时 web”上发言(尽管官方标题是“构建 Leafy Chat、DjangoDose 和 Hurricane，使用 Python 在实时 Web 上的经验教训”)。他谈到的第一个话题是 [Leafy Chat](http://leafychat.com/) ，这是一个在浏览器中使用 [django](http://www.djangoproject.com/) 的聊天程序，他在一次 [django 短跑比赛](http://djangodash.com/)中用 48 小时组建了一个团队。Leafy Chat 使用 JSON 将数据包从客户端传递到服务器(反之亦然)。虽然它可以工作，但是不可扩展。下一个话题是[djangode](http://djangodose.com/)，使用了绕圈、扭转和跺脚。这个 web 应用程序运行得更好，但也不完全稳定，因为它依赖于 Twitter，如果 Twitter 宕机，DjangoDose 也会宕机。

下一个话题是飓风。它使用 jQuery、 [Tornado](http://www.tornadoweb.org/documentation) 服务器和多处理队列来产生和消费数据。它可以提供 twitter feed 和聊天，但问题是应用程序状态完全在内存中。

最后一个题目是 [Redis](http://code.google.com/p/redis/) 。它建立在前面的例子之上，做了许多相同的事情，只是规模更大。幻灯片见 http://us.pycon.org/2010/conference/schedule/event/10/。我真的不明白这和他创造的其他东西有什么不同，但这可能是我的错，

第二个演讲人是丹尼尔·格林菲尔德(又名:[皮丹尼](http://pydanny.blogspot.com/))。我认为他要么是 Pinax 项目的创始人，要么是主要参与者之一。不幸的是，他有严重的技术问题，这使得他的演讲很难继续下去。他甚至没有机会展示任何代码，所以我有点失望。

第一场会议的最后一位演讲者是 Python 核心开发人员 Brett Cannon。它被评为先进，它是。我不太明白他在说什么，所以我很早就离开去吃午饭了。第二次治疗好多了，所以请尽快查看我的帖子。

![](img/2b76168c03a54c548d3dd77a2bc716bb.png)