# Python:坏猴子打补丁的一个例子

> 原文：<https://www.blog.pythonlibrary.org/2016/02/26/python-an-example-of-bad-monkey-patching/>

今天我的一个同事来找我，让我解释他们发现的一些奇怪的 Python 代码。它处理了 cmake，但是因为它是内部代码，所以我不能在这里展示。相反，我写了一些有同样问题的东西，所以你可以看到我认为不好的，或者至少是非常愚蠢的代码:

```py

class Config:

    def Run(self):
        print('Program Usage: blah blah blah')

        print(self.product)

        self.asset_tag = 'test'
        print(self.asset_tag)

        total = self.sub_total_a + self.sub_total_b

```

当我第一次看到它时，我被它如何使用未初始化的属性所震惊，我想知道这是如何工作的。然后我意识到他们一定在做某种猴子补丁。Monkey patching 是您编写代码在运行时动态修改类或模块的地方。所以我进一步查看了脚本，找到了一些代码，它们创建了该类的一个实例，并做了类似这样的事情:

```py

def test_config():
    cfg = Config()
    cfg.product = 'laptop'
    cfg.asset_tag = '12345ABCD'
    cfg.sub_total_a = 10.23
    cfg.sub_total_b = 112.63
    cfg.Run()

if __name__ == '__main__':
    test_config()

```

所以基本上每当你创建一个类的实例时，你都需要修补它，这样在你调用 **Run** 方法之前属性就存在了。

虽然我认为 Python 可以做这种事情很酷，但对于不熟悉 Python 的人来说，这是非常令人困惑的，尤其是当代码像这样缺乏文档记录的时候。如您所见，没有注释或文档字符串，所以需要花点时间来弄清楚发生了什么。幸运的是，所有代码都在同一个文件中。否则这可能会变得非常棘手。

我个人认为这是糟糕编码的一个很好的例子。如果是我写的，我会创建一个 **__init__** 方法，并在那里初始化所有这些属性。那么就不会对这个类的工作方式产生混淆。我也非常相信写好的文档串和有用的注释。

无论如何，我希望你觉得这很有趣。我还认为我的读者应该意识到你将在野外看到的一些奇怪的代码片段。