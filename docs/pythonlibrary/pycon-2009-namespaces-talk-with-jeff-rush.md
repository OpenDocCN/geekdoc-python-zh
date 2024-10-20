# PyCon 2009 -与 Jeff Rush 讨论名称空间

> 原文：<https://www.blog.pythonlibrary.org/2009/03/27/pycon-2009-namespaces-talk-with-jeff-rush/>

我参加的第一个 PyCon 演讲题为 [*关于 Python 名称空间(和代码对象)*](http://us.pycon.org/2009/conference/schedule/event/7/) ，由[杰夫·拉什](http://www.blogger.com/profile/14683745915943062235)主讲。这个演讲对我来说有点难懂。如果 PyCon 组织者发布了幻灯片和视频，一定要抓住它们并观看。快速浏览一下实例化如何创建各种对象，这些对象下面可能有代码对象。他还谈到了如何将变量和函数编译成代码对象。

Rush 还经历了代码对象/名称空间的执行前和执行后阶段。一个有趣的事实是，生成器保持名称空间的时间比普通的本地名称空间长。我不太清楚这在日常编码中意味着什么...但这可能很重要。

他的下一个话题是闭包。下面是他的简单例子:

```py

def salary_above(amt):
def wages_test(person):
return person.salary
return wages_test
```

然后他继续讨论早期的绑定和名称空间。这是他的一个例子:

```py

# Early binding
names = []
rollcall = names.append
rollcall("John")
```

他还谈到了授权和早期绑定，但正如我上面提到的，你最好只看视频和他的幻灯片。我不能像他那样解释清楚。

![](img/b18fe6ef68b2765bd55fe92f7cdde9f2.png)