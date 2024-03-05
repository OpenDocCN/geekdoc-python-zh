# 在 Python 中使用 JSON 数据

> 原文：<https://realpython.com/python-json/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**用 Python 处理 JSON 数据**](/courses/working-json-data-python/)

自从诞生以来， [JSON](https://en.wikipedia.org/wiki/JSON) 已经迅速成为事实上的信息交换标准。很有可能你在这里是因为你需要把一些数据从这里传输到那里。也许你正在通过一个 [API](https://realpython.com/api-integration-in-python/) 收集信息，或者将你的数据存储在一个[文档数据库](https://realpython.com/introduction-to-mongodb-and-python/)中。无论如何，您已经深陷 JSON，您必须使用 Python 才能摆脱困境。

幸运的是，这是一个非常普通的任务，而且——和大多数普通任务一样——Python 让它变得非常简单。别害怕，蟒蛇和蟒蛇们。这次会轻而易举！

> **所以，我们用 JSON 来存储和交换数据？是的，你猜对了！它只不过是社区用来传递数据的标准化格式。请记住，JSON 不是这类工作的唯一可用格式，但是 [XML](https://en.wikipedia.org/wiki/XML) 和 [YAML](https://realpython.com/python-yaml/) 可能是唯一值得一提的格式。**

**免费 PDF 下载:** [Python 3 备忘单](https://realpython.com/bonus/python-cheat-sheet-short/)

## JSON 的(非常)简史

不足为奇的是，**J**ava**S**script**O**object**N**rotation 的灵感来自于处理对象字面语法的 [JavaScript 编程语言](https://realpython.com/python-vs-javascript/)的子集。他们有一个漂亮的网站来解释整个事情。不过不要担心:JSON 早已成为语言不可知论者，并作为自己的标准而存在，所以我们可以出于讨论的目的而避开 JavaScript。

最终，整个社区都采用了 JSON，因为它易于人类和机器创建和理解。

[*Remove ads*](/account/join/)

## 看，是 JSON！

准备好。我将向您展示一些真实的 JSON——就像您在野外看到的一样。没关系:JSON 应该是任何使用过 C 风格语言的人都可读的，而 [Python 是一种 C 风格语言](https://realpython.com/c-for-python-programmers/)…所以那就是你！

```py
{ "firstName":  "Jane", "lastName":  "Doe", "hobbies":  ["running",  "sky diving",  "singing"], "age":  35, "children":  [ { "firstName":  "Alice", "age":  6 }, { "firstName":  "Bob", "age":  8 } ] }
```

如您所见，JSON 支持基本类型，如[字符串](https://realpython.com/python-strings/)和[数字](https://realpython.com/python-numbers/)，以及嵌套列表和对象。

> 等等，那看起来像一本 Python 字典！我知道，对吧？在这一点上，它几乎是通用的对象符号，但我不认为 UON 能很好地脱口而出。欢迎在评论中讨论替代方案。

咻！你在第一次遭遇野生 JSON 时幸存了下来。现在你只需要学会如何驯服它。

## Python 原生支持 JSON！

Python 自带了一个名为 [`json`](https://docs.python.org/3/library/json.html) 的内置包，用于编码和解码 JSON 数据。

把这个小家伙放在你档案的最上面:

```py
import json
```

### 一点词汇

对 JSON 进行编码的过程通常被称为**序列化**。这个术语指的是将数据转换成一个*字节序列*(因此成为*序列*)以便存储或通过网络传输。你可能也听说过术语**编组**，但那是[一个完全不同的讨论](https://stackoverflow.com/questions/770474/what-is-the-difference-between-serialization-and-marshaling)。自然地，**反序列化**是解码已经以 JSON 标准存储或交付的数据的相反过程。

> 哎呀！这听起来很专业。肯定。但实际上，我们在这里谈论的都是*读*和*写*。可以这样想:*编码*是为了*将*数据写入磁盘，而*解码*是为了*将*数据读入内存。

### 正在序列化 JSON

计算机处理大量信息后会发生什么？它需要进行数据转储。相应地，`json`库公开了将数据写入文件的`dump()`方法。还有一个用于写入 Python 字符串的`dumps()`方法(读作“dump-s”)。

简单的 Python 对象根据一种相当直观的转换被翻译成 JSON。

| 计算机编程语言 | JSON |
| --- | --- |
| `dict` | `object` |
| `list`，`tuple` | `array` |
| `str` | `string` |
| `int`、`long`、`float` | `number` |
| `True` | `true` |
| `False` | `false` |
| `None` | `null` |

### 一个简单的序列化例子

假设您正在内存中处理一个 Python 对象，看起来有点像这样:

```py
data = {
    "president": {
        "name": "Zaphod Beeblebrox",
        "species": "Betelgeusian"
    }
}
```

将这些信息保存到磁盘上是非常重要的，因此您的任务是将其写入文件。

使用 Python 的上下文管理器，您可以创建一个名为`data_file.json`的文件，并以写模式打开它。(JSON 文件通常以扩展名`.json`结尾。)

```py
with open("data_file.json", "w") as write_file:
    json.dump(data, write_file)
```

注意`dump()`有两个位置参数:(1)要序列化的数据对象，以及(2)字节将被写入的类似文件的对象。

或者，如果您倾向于在程序中继续使用这种序列化的 JSON 数据，您可以将它写入一个本机 Python `str`对象。

```py
json_string = json.dumps(data)
```

请注意，类似文件的对象不存在，因为您实际上没有写入磁盘。除此之外，`dumps()`就跟`dump()`一样。

万岁！你生了一些小 JSON，你准备把它放回野外，让它长得又大又壮。

[*Remove ads*](/account/join/)

### 一些有用的关键字参数

请记住，JSON 是为了让人们容易阅读，但是如果把所有的语法都挤在一起，可读的语法是不够的。另外，你可能有一种不同于我的编程风格，当代码被格式化成你喜欢的格式时，你可能更容易阅读它。

> **注意:**`dump()`和`dumps()`方法使用相同的关键字参数。

大多数人想要改变的第一个选项是空白。您可以使用`indent`关键字参数来指定嵌套结构的缩进大小。通过使用我们在上面定义的`data`，并在控制台中运行以下命令，来亲自检查一下不同之处:

>>>

```py
>>> json.dumps(data)
>>> json.dumps(data, indent=4)
```

另一个格式化选项是`separators`关键字参数。默认情况下，这是一个 2 元组的分隔符字符串`(", ", ": ")`，但是 compact JSON 的一个常见替代字符串是`(",", ":")`。再次查看示例 JSON，看看这些分隔符在哪里发挥作用。

还有其他的，比如`sort_keys`，但是我不知道那个是做什么的。如果你好奇的话，你可以在[文档](https://docs.python.org/3/library/json.html#basic-usage)中找到完整的列表。

### 反序列化 JSON

太好了，看起来你已经为自己捕获了一些野生 JSON！现在是时候让它成形了。在`json`库中，您会发现用于将 JSON 编码的数据转换成 Python 对象的`load()`和`loads()`。

就像序列化一样，反序列化也有一个简单的转换表，不过您可能已经猜到它是什么样子了。

| JSON | 计算机编程语言 |
| --- | --- |
| `object` | `dict` |
| `array` | `list` |
| `string` | `str` |
| `number`(整数) | `int` |
| `number`(真实) | `float` |
| `true` | `True` |
| `false` | `False` |
| `null` | `None` |

从技术上讲，这种转换并不是序列化表的完美逆过程。这基本上意味着，如果你现在对一个对象进行编码，然后再解码，你可能得不到完全相同的对象。我想象这有点像传送:在这里分解我的分子，然后在那里把它们重新组合起来。我还是原来的我吗？

在现实中，这可能更像是让一个朋友把一些东西翻译成日语，另一个朋友把它翻译回英语。不管怎样，最简单的例子是对一个 [`tuple`](https://realpython.com/python-lists-tuples/) 进行编码，并在解码后得到一个`list`，就像这样:

>>>

```py
>>> blackjack_hand = (8, "Q")
>>> encoded_hand = json.dumps(blackjack_hand)
>>> decoded_hand = json.loads(encoded_hand)

>>> blackjack_hand == decoded_hand
False
>>> type(blackjack_hand)
<class 'tuple'>
>>> type(decoded_hand)
<class 'list'>
>>> blackjack_hand == tuple(decoded_hand)
True
```

### 一个简单的反序列化示例

这一次，假设您已经在磁盘上存储了一些数据，您希望在内存中操作这些数据。您仍将使用上下文管理器，但这次您将以读取模式打开现有的`data_file.json`。

```py
with open("data_file.json", "r") as read_file:
    data = json.load(read_file)
```

这里的事情非常简单，但是请记住，这个方法的结果可以从转换表中返回任何允许的[数据类型](https://realpython.com/python-data-types/)。这只有在你加载以前没有见过的数据时才重要。在大多数情况下，根对象将是一个`dict`或一个`list`。

如果您已经从另一个程序获取了 JSON 数据，或者以其他方式获得了 Python 中的一串 JSON 格式的数据，那么您可以很容易地用`loads()`对其进行反序列化，它自然地从一个字符串中加载:

```py
json_string = """
{
 "researcher": {
 "name": "Ford Prefect",
 "species": "Betelgeusian",
 "relatives": [
 {
 "name": "Zaphod Beeblebrox",
 "species": "Betelgeusian"
 }
 ]
 }
}
"""
data = json.loads(json_string)
```

瞧啊。你驯服了野 JSON，现在它在你的控制之下。但是你用这种力量做什么取决于你自己。你可以喂养它，培养它，甚至教它一些技巧。我不是不信任你…但是要控制好它，好吗？

[*Remove ads*](/account/join/)

## 一个真实世界的例子

对于您的介绍性示例，您将使用 [JSONPlaceholder](https://jsonplaceholder.typicode.com/) ，这是一个用于实践目的的假 JSON 数据源。

首先创建一个名为`scratch.py`的脚本文件，或者你想要的任何东西。我真的不能阻止你。

您需要向 JSONPlaceholder 服务发出一个 API 请求，所以只需使用 [`requests`](http://docs.python-requests.org/en/master/) 包来完成繁重的工作。将这些导入添加到文件的顶部:

```py
import json
import requests
```

现在，你将会处理一份待办事项清单，因为就像…你知道，这是一种通过仪式或其他什么。

继续向 JSONPlaceholder API 请求`/todos`端点。如果您不熟悉`requests`，实际上有一个方便的`json()`方法可以为您完成所有工作，但是您可以练习使用`json`库来反序列化响应对象的`text`属性。它应该是这样的:

```py
response = requests.get("https://jsonplaceholder.typicode.com/todos")
todos = json.loads(response.text)
```

你不相信这有用吗？好吧，在交互模式下运行文件，自己测试一下。同时，检查一下`todos`的类型。如果你想冒险，看看列表中的前 10 个项目。

>>>

```py
>>> todos == response.json()
True
>>> type(todos)
<class 'list'>
>>> todos[:10]
...
```

我不会骗你，但我很高兴你是个怀疑论者。

> **什么是交互模式？**啊，我还以为你不会问呢！你知道你总是在编辑器和终端之间跳来跳去吗？嗯，我们这些狡猾的 python 爱好者在运行脚本时使用了`-i`交互标志。这是测试代码的一个很棒的小技巧，因为它运行脚本，然后打开一个交互式命令提示符，可以访问脚本中的所有数据！

好了，该行动了。您可以通过在浏览器中访问[端点](https://jsonplaceholder.typicode.com/todos)来查看数据的结构，但是这里有一个示例 TODO:

```py
{ "userId":  1, "id":  1, "title":  "delectus aut autem", "completed":  false }
```

有多个用户，每个用户都有一个惟一的`userId`，每个任务都有一个布尔`completed`属性。您能确定哪些用户完成了最多的任务吗？

```py
# Map of userId to number of complete TODOs for that user
todos_by_user = {}

# Increment complete TODOs count for each user.
for todo in todos:
    if todo["completed"]:
        try:
            # Increment the existing user's count.
            todos_by_user[todo["userId"]] += 1
        except KeyError:
            # This user has not been seen. Set their count to 1.
            todos_by_user[todo["userId"]] = 1

# Create a sorted list of (userId, num_complete) pairs.
top_users = sorted(todos_by_user.items(), 
                   key=lambda x: x[1], reverse=True)

# Get the maximum number of complete TODOs.
max_complete = top_users[0][1]

# Create a list of all users who have completed
# the maximum number of TODOs.
users = []
for user, num_complete in top_users:
    if num_complete < max_complete:
        break
    users.append(str(user))

max_users = " and ".join(users)
```

是的，是的，您的实现更好，但关键是，您现在可以像操作普通 Python 对象一样操作 JSON 数据！

我不知道您是怎么想的，但是当我再次以交互方式运行该脚本时，我会得到以下结果:

>>>

```py
>>> s = "s" if len(users) > 1 else ""
>>> print(f"user{s}  {max_users} completed {max_complete} TODOs")
users 5 and 10 completed 12 TODOs
```

这很酷，但你是来学习 JSON 的。对于您的最后一个任务，您将创建一个 JSON 文件，其中包含每个完成了最多待办事项的用户的已完成待办事项。

您所需要做的就是过滤`todos`并将结果列表写入一个文件。出于原创的考虑，可以调用输出文件`filtered_data_file.json`。有很多方法可以做到这一点，但这里有一个:

```py
# Define a function to filter out completed TODOs 
# of users with max completed TODOS.
def keep(todo):
    is_complete = todo["completed"]
    has_max_count = str(todo["userId"]) in users
    return is_complete and has_max_count

# Write filtered TODOs to file.
with open("filtered_data_file.json", "w") as data_file:
    filtered_todos = list(filter(keep, todos))
    json.dump(filtered_todos, data_file, indent=2)
```

太好了，你已经去掉了所有你不需要的数据，把好的东西保存到了一个全新的文件中！再次运行脚本并检查`filtered_data_file.json`以验证一切正常。当您运行它时，它将与`scratch.py`在同一个目录中。

既然你已经走了这么远，我打赌你一定感觉很棒，对吧？不要骄傲自大:谦逊是一种美德。不过，我倾向于同意你的观点。到目前为止，这是一帆风顺的，但你可能要为这最后一段旅程做好准备。

[*Remove ads*](/account/join/)

## 编码和解码自定义 Python 对象

当我们试图从你正在开发的地下城&龙应用中序列化`Elf`类时会发生什么？

```py
class Elf:
    def __init__(self, level, ability_scores=None):
        self.level = level
        self.ability_scores = {
            "str": 11, "dex": 12, "con": 10,
            "int": 16, "wis": 14, "cha": 13
        } if ability_scores is None else ability_scores
        self.hp = 10 + self.ability_scores["con"]
```

不足为奇的是，Python 抱怨说`Elf`不是*可序列化的*(如果你曾经试图告诉一个小精灵，你就会知道这一点):

>>>

```py
>>> elf = Elf(level=4)
>>> json.dumps(elf)
TypeError: Object of type 'Elf' is not JSON serializable
```

虽然`json`模块可以处理大多数内置的 Python 类型，但是它并不理解默认情况下如何编码定制的数据类型。这就像试图把一个方钉装进一个圆孔——你需要一个电锯和父母的监督。

### 简化数据结构

现在，问题是如何处理更复杂的数据结构。嗯，您可以尝试手工编码和解码 JSON，但是有一个稍微聪明一点的解决方案可以帮您节省一些工作。您可以插入一个中间步骤，而不是直接从定制数据类型转换到 JSON。

你所需要做的就是用`json`已经理解的内置类型来表示你的数据。本质上，您将更复杂的对象转换成更简单的表示，然后由`json`模块转换成 JSON。这就像数学中的传递性:如果 A = B and B = C，那么 A = C

要掌握这一点，你需要一个复杂的对象来玩。您可以使用任何您喜欢的自定义类，但是 Python 有一个名为`complex`的内置类型，用于表示[复数](https://realpython.com/python-complex-numbers/)，默认情况下它是不可序列化的。因此，为了这些例子，你的复杂对象将是一个`complex`对象。困惑了吗？

>>>

```py
>>> z = 3 + 8j
>>> type(z)
<class 'complex'>
>>> json.dumps(z)
TypeError: Object of type 'complex' is not JSON serializable
```

> **复数从何而来？**你看，当一个实数和一个虚数非常相爱时，它们加在一起产生一个数，这个数(名正言顺地)叫做 [*复数*](https://www.mathsisfun.com/numbers/complex-numbers.html) 。

当使用自定义类型时，一个很好的问题是**重新创建这个对象所需的最少信息量是多少？**在复数的情况下，你只需要知道实部和虚部，这两部分都可以作为属性在`complex`对象上访问:

>>>

```py
>>> z.real
3.0
>>> z.imag
8.0
```

将相同的数字传递给`complex`构造函数足以满足`__eq__`比较运算符:

>>>

```py
>>> complex(3, 8) == z
True
```

将自定义数据类型分解成基本组件对于序列化和反序列化过程都至关重要。

### 编码自定义类型

要将一个定制对象翻译成 JSON，您需要做的就是为`dump()`方法的`default`参数提供一个编码函数。`json`模块将在任何非本地可序列化的对象上调用这个函数。这里有一个简单的解码函数，你可以用来练习:

```py
def encode_complex(z):
    if isinstance(z, complex):
        return (z.real, z.imag)
    else:
        type_name = z.__class__.__name__
        raise TypeError(f"Object of type '{type_name}' is not JSON serializable")
```

请注意，如果您没有得到所期望的那种对象，那么您应该抛出一个`TypeError`。这样，您就避免了意外序列化任何精灵。现在您可以自己尝试编码复杂的对象了！

>>>

```py
>>> json.dumps(9 + 5j, default=encode_complex)
'[9.0, 5.0]'
>>> json.dumps(elf, default=encode_complex)
TypeError: Object of type 'Elf' is not JSON serializable
```

> **为什么我们把复数编码成一个`tuple`？**好问题！这当然不是唯一的选择，也不一定是最好的选择。事实上，如果您以后想要解码该对象，这不是一个很好的表示，您很快就会看到这一点。

另一种常见的方法是子类化标准的`JSONEncoder`并覆盖它的`default()`方法:

```py
class ComplexEncoder(json.JSONEncoder):
    def default(self, z):
        if isinstance(z, complex):
            return (z.real, z.imag)
        else:
            return super().default(z)
```

您可以简单地让基类处理它，而不是自己引发`TypeError`。您可以通过`cls`参数直接在`dump()`方法中使用它，或者通过创建一个编码器实例并调用它的`encode()`方法来使用它:

>>>

```py
>>> json.dumps(2 + 5j, cls=ComplexEncoder)
'[2.0, 5.0]'

>>> encoder = ComplexEncoder()
>>> encoder.encode(3 + 6j)
'[3.0, 6.0]'
```

[*Remove ads*](/account/join/)

### 解码自定义类型

虽然复数的实部和虚部是绝对必要的，但它们实际上并不足以重建物体。当您尝试用`ComplexEncoder`对一个复数进行编码，然后对结果进行解码时，就会发生这种情况:

>>>

```py
>>> complex_json = json.dumps(4 + 17j, cls=ComplexEncoder)
>>> json.loads(complex_json)
[4.0, 17.0]
```

您得到的只是一个列表，如果您还想要那个复杂的对象，您必须将值传递给一个`complex`构造函数。回忆一下我们关于[传送](#deserializing-json)的讨论。缺少的是*元数据*，或者关于你正在编码的数据类型的信息。

我想你真正应该问自己的问题是**重建这个物体所需的*必要的*和*足够的*的最小信息量是多少？**

在 JSON 标准中，`json`模块期望所有的自定义类型都表示为`objects`。为了多样化，这次您可以创建一个名为`complex_data.json`的 JSON 文件，并添加下面的`object`来表示一个复数:

```py
{ "__complex__":  true, "real":  42, "imag":  36 }
```

看到聪明的地方了吗？那个`"__complex__"`键就是我们刚刚谈到的元数据。关联值是多少并不重要。要让这个小技巧发挥作用，您需要做的就是验证密钥是否存在:

```py
def decode_complex(dct):
    if "__complex__" in dct:
        return complex(dct["real"], dct["imag"])
    return dct
```

如果`"__complex__"`不在[字典](https://realpython.com/python-dicts/)中，你可以返回对象，让默认的解码器处理它。

每次`load()`方法试图解析`object`时，您都有机会在默认解码器处理数据之前进行调解。您可以通过将解码函数传递给`object_hook`参数来实现这一点。

现在玩和以前一样的游戏:

>>>

```py
>>> with open("complex_data.json") as complex_data:
...     data = complex_data.read()
...     z = json.loads(data, object_hook=decode_complex)
... 
>>> type(z)
<class 'complex'>
```

虽然`object_hook`可能感觉像是`dump()`方法的`default`参数的对应物，但是这种类比实际上是从这里开始和结束的。

这也不仅仅适用于一个对象。尝试将这个复数列表放入`complex_data.json`并再次运行脚本:

```py
[ { "__complex__":true, "real":42, "imag":36 }, { "__complex__":true, "real":64, "imag":11 } ]
```

如果一切顺利，您将获得一个`complex`对象列表:

>>>

```py
>>> with open("complex_data.json") as complex_data:
...     data = complex_data.read()
...     numbers = json.loads(data, object_hook=decode_complex)
... 
>>> numbers
[(42+36j), (64+11j)]
```

您也可以尝试子类化`JSONDecoder`并覆盖`object_hook`，但是最好尽可能坚持使用轻量级解决方案。

## 全部完成！

恭喜你，现在你可以运用 JSON 的强大力量来满足你所有的邪恶的 Python 需求了。

虽然您在这里使用的示例肯定是人为的并且过于简单，但是它们展示了一个您可以应用于更一般任务的工作流:

1.  [导入](https://realpython.com/absolute-vs-relative-python-imports/)的`json`包。
2.  用`load()`或`loads()`读取数据。
3.  处理数据。
4.  用`dump()`或`dumps()`写入更改的数据。

一旦数据被加载到内存中，您将如何处理它取决于您的用例。一般来说，你的目标是从一个来源收集数据，提取有用的信息，并将这些信息传递下去或记录下来。

今天你进行了一次旅行:你捕获并驯服了一些野生 JSON，并及时赶回来吃晚饭！作为一个额外的奖励，学习`json`包将使学习 [`pickle`](https://realpython.com/python-pickle-module/) 和 [`marshal`](https://docs.python.org/3/library/marshal.html) 变得轻而易举。

祝你在未来的 Pythonic 努力中好运！

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**用 Python 处理 JSON 数据**](/courses/working-json-data-python/)*******