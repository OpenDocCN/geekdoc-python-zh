# 工厂方法模式及其在 Python 中的实现

> 原文：<https://realpython.com/factory-method-python/>

本文探索工厂方法设计模式及其在 Python 中的实现。在所谓的四人组(GoF: Gamma、Helm、Johson 和 Vlissides)出版了他们的书[Design Patterns:Elements of Reusable Object-Oriented Software](https://realpython.com/asins/0201633612/)之后，设计模式在 90 年代后期成为了一个热门话题。

这本书将设计模式描述为解决软件中反复出现的问题的核心设计方案，并根据问题的性质将每个设计模式分为[类](https://en.wikipedia.org/wiki/Software_design_pattern#Classification_and_list)。每个模式都有一个名称、一个问题描述、一个设计解决方案，以及对使用它的后果的解释。

GoF 书将工厂方法描述为一种创造性的设计模式。创建性设计模式与对象的创建相关，工厂方法是一种使用公共[接口](https://realpython.com/python-interface/)创建对象的设计模式。

这是一个反复出现的问题**使得工厂方法成为最广泛使用的设计模式之一**，理解它并知道如何应用它是非常重要的。

**本文结束时，您将**:

*   理解工厂方法的组成部分
*   识别在应用程序中使用工厂方法的机会
*   学习使用该模式修改现有代码并改进其设计
*   学会识别工厂方法是合适的设计模式的机会
*   选择适当的工厂方法实现
*   知道如何实现工厂方法的可重用、通用的解决方案

**免费奖励:** [掌握 Python 的 5 个想法](https://realpython.com/bonus/python-mastery-course/)，这是一个面向 Python 开发者的免费课程，向您展示将 Python 技能提升到下一个水平所需的路线图和心态。

## 介绍工厂方法

工厂方法是一种创造性的设计模式，用于创建公共接口的具体实现。

它将创建对象的过程与依赖于对象接口的代码分开。

例如，应用程序需要一个具有特定接口的对象来执行其任务。接口的具体实现由一些参数来标识。

应用程序没有使用复杂的`if/elif/else`条件结构来决定具体的实现，而是将这个决定委托给一个创建具体对象的独立组件。使用这种方法，应用程序代码得到了简化，使其更易于重用和维护。

假设一个应用程序需要使用指定的格式将一个`Song`对象转换成它的 [`string`](https://realpython.com/python-strings/) 表示。将对象转换为不同的表示形式通常称为序列化。您经常会看到这些需求在包含所有逻辑和实现的单个函数或方法中实现，如下面的代码所示:

```py
# In serializer_demo.py

import json
import xml.etree.ElementTree as et

class Song:
    def __init__(self, song_id, title, artist):
        self.song_id = song_id
        self.title = title
        self.artist = artist

class SongSerializer:
    def serialize(self, song, format):
        if format == 'JSON':
            song_info = {
                'id': song.song_id,
                'title': song.title,
                'artist': song.artist
            }
            return json.dumps(song_info)
        elif format == 'XML':
            song_info = et.Element('song', attrib={'id': song.song_id})
            title = et.SubElement(song_info, 'title')
            title.text = song.title
            artist = et.SubElement(song_info, 'artist')
            artist.text = song.artist
            return et.tostring(song_info, encoding='unicode')
        else:
            raise ValueError(format)
```

在上面的例子中，有一个基本的`Song`类来表示一首歌，还有一个`SongSerializer`类可以根据`format`参数的值将`song`对象转换成它的`string`表示。

`.serialize()`方法支持两种不同的格式: [JSON](https://json.org/) 和 [XML](https://www.xml.com/axml/axml.html) 。任何其他指定的`format`都不被支持，因此会引发一个`ValueError`异常。

让我们使用 Python 交互式 shell 来看看代码是如何工作的:

>>>

```py
>>> import serializer_demo as sd
>>> song = sd.Song('1', 'Water of Love', 'Dire Straits')
>>> serializer = sd.SongSerializer()

>>> serializer.serialize(song, 'JSON')
'{"id": "1", "title": "Water of Love", "artist": "Dire Straits"}'

>>> serializer.serialize(song, 'XML')
'<song id="1"><title>Water of Love</title><artist>Dire Straits</artist></song>'

>>> serializer.serialize(song, 'YAML')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "./serializer_demo.py", line 30, in serialize
    raise ValueError(format)
ValueError: YAML
```

您创建了一个`song`对象和一个`serializer`，并通过使用`.serialize()`方法将歌曲转换成它的字符串表示。该方法将`song`对象作为参数，以及一个表示所需格式的字符串值。最后一个调用使用`YAML`作为格式，`serializer`不支持，因此引发了`ValueError`异常。

这个例子简短而简化，但是仍然很复杂。根据`format`参数的值，有三种逻辑或执行路径。这看起来没什么大不了的，你可能见过比这更复杂的代码，但是上面的例子仍然很难维护。

[*Remove ads*](/account/join/)

### 复杂条件代码的问题

上面的例子展示了你会在复杂的逻辑代码中发现的所有问题。复杂的逻辑代码使用`if/elif/else`结构来改变应用程序的行为。使用`if/elif/else`条件结构使得代码更难阅读、理解和维护。

上面的代码可能看起来不难阅读或理解，但是请等到您看到本节中的最终代码！

然而，上面的代码很难维护，因为它做的太多了。[单一责任原则](https://en.wikipedia.org/wiki/Single_responsibility_principle)声明一个[模块](https://realpython.com/python-modules-packages/)，一个类，甚至一个方法应该有一个单一的、定义明确的责任。它应该只做一件事，并且只有一个改变的理由。

由于许多不同的原因，`SongSerializer`中的`.serialize()`方法将需要更改。这增加了引入新缺陷或破坏现有功能的风险。让我们看一下需要修改实施的所有情况:

*   当一种新的格式被引入时:方法将不得不改变以实现对该格式的序列化。

*   **当`Song`对象改变时:**向`Song`类添加或移除属性将需要实现改变以适应新的结构。

*   **当格式的字符串表示发生变化时(普通的[JSON](https://json.org/)vs[JSON API](https://jsonapi.org/)):**如果格式所需的字符串表示发生变化，那么`.serialize()`方法也必须发生变化，因为该表示在`.serialize()`方法实现中是硬编码的。

理想的情况是，在不改变`.serialize()`方法的情况下，可以实现需求中的任何变化。让我们在接下来的几节中看看如何做到这一点。

### 寻找通用接口

当您在应用程序中看到复杂的条件代码时，第一步是确定每个执行路径(或逻辑路径)的共同目标。

使用`if/elif/else`的代码通常有一个共同的目标，在每个逻辑路径中以不同的方式实现。上面的代码在每个逻辑路径中使用不同的格式将一个`song`对象转换成它的`string`表示。

基于这个目标，你寻找一个公共接口来替换每一条路径。上面的例子需要一个接口，它接受一个`song`对象并返回一个`string`。

一旦有了公共接口，就可以为每个逻辑路径提供单独的实现。在上面的例子中，您将提供一个序列化为 JSON 的实现和另一个序列化为 XML 的实现。

然后，您提供一个单独的组件，它根据指定的`format`决定要使用的具体实现。该组件评估`format`的值，并返回由其值标识的具体实现。

在下面几节中，您将学习如何在不改变行为的情况下对现有代码进行更改。这被称为[重构](https://en.wikipedia.org/wiki/Code_refactoring)代码。

Martin Fowler 在他的书[Refactoring:Improving the Design of Existing Code](https://realpython.com/asins/0134757599/)中将重构定义为“以不改变代码的外部行为但改善其内部结构的方式改变软件系统的过程。”如果你想看看重构的实际操作，看看真正的 Python 代码对话[重构:准备你的代码以获得帮助](https://realpython.com/courses/refactoring-code-to-get-help/)。

让我们开始重构代码，以获得使用工厂方法设计模式的所需结构。

### 将代码重构为所需的界面

所需的接口是一个对象或函数，它接受一个`Song`对象并返回一个`string`表示。

第一步是将其中一个逻辑路径重构到这个接口中。您可以通过添加一个新方法`._serialize_to_json()`并将 JSON 序列化代码移动到其中来实现这一点。然后，您更改客户端来调用它，而不是在 [`if`语句](https://realpython.com/python-conditional-statements/)的主体中实现它:

```py
class SongSerializer:
    def serialize(self, song, format):
        if format == 'JSON':
            return self._serialize_to_json(song)
        # The rest of the code remains the same

    def _serialize_to_json(self, song):
        payload = {
            'id': song.song_id,
            'title': song.title,
            'artist': song.artist
        }
        return json.dumps(payload)
```

一旦进行了这种更改，您就可以验证行为是否没有改变。然后，对 XML 选项做同样的事情，引入一个新方法`._serialize_to_xml()`，将实现移到它上面，并修改`elif`路径来调用它。

以下示例显示了重构后的代码:

```py
class SongSerializer:
    def serialize(self, song, format):
        if format == 'JSON':
            return self._serialize_to_json(song)
        elif format == 'XML':
            return self._serialize_to_xml(song)
        else:
            raise ValueError(format)

    def _serialize_to_json(self, song):
        payload = {
            'id': song.song_id,
            'title': song.title,
            'artist': song.artist
        }
        return json.dumps(payload)

    def _serialize_to_xml(self, song):
        song_element = et.Element('song', attrib={'id': song.song_id})
        title = et.SubElement(song_element, 'title')
        title.text = song.title
        artist = et.SubElement(song_element, 'artist')
        artist.text = song.artist
        return et.tostring(song_element, encoding='unicode')
```

新版本的代码更容易阅读和理解，但仍然可以通过工厂方法的基本实现进行改进。

[*Remove ads*](/account/join/)

### 工厂方法的基本实现

工厂方法的中心思想是提供一个独立的组件，负责根据一些指定的参数来决定应该使用哪个具体的实现。我们示例中的参数是`format`。

为了完成工厂方法的实现，您添加了一个新方法`._get_serializer()`，它采用了所需的`format`。该方法计算`format`的值，并返回匹配的序列化函数:

```py
class SongSerializer:
    def _get_serializer(self, format):
        if format == 'JSON':
            return self._serialize_to_json
        elif format == 'XML':
            return self._serialize_to_xml
        else:
            raise ValueError(format)
```

**注意:**`._get_serializer()`方法不调用具体的实现，它只是返回函数对象本身。

现在，您可以将`SongSerializer`的`.serialize()`方法改为使用`._get_serializer()`来完成工厂方法实现。下一个示例显示了完整的代码:

```py
class SongSerializer:
    def serialize(self, song, format):
        serializer = self._get_serializer(format)
        return serializer(song)

    def _get_serializer(self, format):
        if format == 'JSON':
            return self._serialize_to_json
        elif format == 'XML':
            return self._serialize_to_xml
        else:
            raise ValueError(format)

    def _serialize_to_json(self, song):
        payload = {
            'id': song.song_id,
            'title': song.title,
            'artist': song.artist
        }
        return json.dumps(payload)

    def _serialize_to_xml(self, song):
        song_element = et.Element('song', attrib={'id': song.song_id})
        title = et.SubElement(song_element, 'title')
        title.text = song.title
        artist = et.SubElement(song_element, 'artist')
        artist.text = song.artist
        return et.tostring(song_element, encoding='unicode')
```

最终的实现展示了工厂方法的不同组件。`.serialize()`方法是依赖一个接口来完成其任务的应用程序代码。

这被称为模式的**客户端**组件。定义的接口被称为**产品**组件。在我们的例子中，产品是一个函数，它接受一个`Song`并返回一个字符串表示。

`._serialize_to_json()`和`._serialize_to_xml()`方法是产品的具体实现。最后，`._get_serializer()`方法是**创建者**组件。创建者决定使用哪个具体的实现。

因为您是从一些现有代码开始的，所以 Factory Method 的所有组件都是同一个类`SongSerializer`的成员。

通常情况并非如此，正如您所见，添加的方法都不使用`self`参数。这很好地表明它们不应该是`SongSerializer`类的方法，它们可以成为外部函数:

```py
class SongSerializer:
    def serialize(self, song, format):
        serializer = get_serializer(format)
        return serializer(song)

def get_serializer(format):
    if format == 'JSON':
        return _serialize_to_json
    elif format == 'XML':
        return _serialize_to_xml
    else:
        raise ValueError(format)

def _serialize_to_json(song):
    payload = {
        'id': song.song_id,
        'title': song.title,
        'artist': song.artist
    }
    return json.dumps(payload)

def _serialize_to_xml(song):
    song_element = et.Element('song', attrib={'id': song.song_id})
    title = et.SubElement(song_element, 'title')
    title.text = song.title
    artist = et.SubElement(song_element, 'artist')
    artist.text = song.artist
    return et.tostring(song_element, encoding='unicode')
```

**注意:**`SongSerializer`中的`.serialize()`方法不使用`self`参数。

上面的规则告诉我们它不应该是类的一部分。这是正确的，但是您处理的是现有的代码。

如果您删除了`SongSerializer`并将`.serialize()`方法改为一个函数，那么您必须更改应用程序中使用`SongSerializer`的所有位置，并替换对新函数的调用。

除非你的单元测试有很高的代码覆盖率，否则这不是你应该做的改变。

工厂方法的机制总是相同的。客户端(`SongSerializer.serialize()`)依赖于接口的具体实现。它使用某种标识符(`format`)向创建者组件(`get_serializer()`)请求实现。

创建者根据参数的值将具体实现返回给客户端，客户端使用提供的对象完成其任务。

您可以在 Python 交互式解释器中执行相同的指令集，以验证应用程序行为没有改变:

>>>

```py
>>> import serializer_demo as sd
>>> song = sd.Song('1', 'Water of Love', 'Dire Straits')
>>> serializer = sd.SongSerializer()

>>> serializer.serialize(song, 'JSON')
'{"id": "1", "title": "Water of Love", "artist": "Dire Straits"}'

>>> serializer.serialize(song, 'XML')
'<song id="1"><title>Water of Love</title><artist>Dire Straits</artist></song>'

>>> serializer.serialize(song, 'YAML')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "./serializer_demo.py", line 13, in serialize
    serializer = get_serializer(format)
  File "./serializer_demo.py", line 23, in get_serializer
    raise ValueError(format)
ValueError: YAML
```

您创建了一个`song`和一个`serializer`，并使用`serializer`将歌曲转换为其指定了一个`format`的`string`表示。由于 [`YAML`](https://realpython.com/python-yaml/) 不是受支持的格式，因此引发`ValueError`。

## 识别使用工厂方法的机会

工厂方法应该用在应用程序(客户端)依赖于接口(产品)来执行任务，并且该接口有多个具体实现的每种情况下。你需要提供一个可以标识具体实现的参数，并在 creator 中使用它来决定具体的实现。

符合这种描述的问题范围很广，所以让我们看一些具体的例子。

**替换复杂的逻辑代码:**`if/elif/else`格式的复杂逻辑结构很难维护，因为随着需求的变化，需要新的逻辑路径。

工厂方法是一个很好的替代方法，因为您可以将每个逻辑路径的主体放入具有公共接口的独立函数或类中，并且创建者可以提供具体的实现。

在条件中评估的参数成为识别具体实现的参数。上面的例子代表了这种情况。

**从外部数据构建相关对象:**假设一个应用程序需要从数据库或其他外部来源检索员工信息。

这些记录代表不同角色或类型的雇员:经理、办公室职员、销售助理等等。应用程序可以在记录中存储一个代表雇员类型的标识符，然后使用工厂方法从记录中的其余信息创建每个具体的`Employee`对象。

**支持同一功能的多种实现:**一个图像处理应用程序需要将卫星图像从一个坐标系转换到另一个坐标系，但是有多种不同精度级别的算法来执行转换。

应用程序可以允许用户选择识别具体算法的选项。工厂方法可以提供基于该选项的算法的具体实现。

**在公共接口下组合相似的特征:**在图像处理示例之后，应用程序需要对图像应用滤镜。要使用的特定过滤器可以通过一些用户输入来识别，工厂方法可以提供具体的过滤器实现。

**集成相关的外部服务:**一个音乐播放器应用程序想要集成多个外部服务，并允许用户选择他们的音乐来自哪里。应用程序可以为音乐服务定义一个公共接口，并使用工厂方法根据用户偏好创建正确的集成。

这些情况都差不多。它们都定义了一个客户端，该客户端依赖于一个称为产品的公共接口。它们都提供了识别产品具体实现的方法，所以它们都可以在设计中使用工厂方法。

现在，您可以从前面的示例中了解序列化问题，并通过考虑工厂方法设计模式来提供更好的设计。

[*Remove ads*](/account/join/)

### 一个对象序列化的例子

上例的基本要求是您希望将`Song`对象序列化为它们的`string`表示。这个应用程序似乎提供了与音乐相关的特性，所以这个应用程序可能需要序列化其他类型的对象，比如`Playlist`或`Album`。

理想情况下，设计应该支持通过实现新类来为新对象添加序列化，而不需要对现有实现进行更改。应用程序需要将对象序列化为多种格式，如 JSON 和 XML，因此定义一个可以有多种实现的接口`Serializer`似乎是很自然的，每种格式一个实现。

接口实现可能如下所示:

```py
# In serializers.py

import json
import xml.etree.ElementTree as et

class JsonSerializer:
    def __init__(self):
        self._current_object = None

    def start_object(self, object_name, object_id):
        self._current_object = {
            'id': object_id
        }

    def add_property(self, name, value):
        self._current_object[name] = value

    def to_str(self):
        return json.dumps(self._current_object)

class XmlSerializer:
    def __init__(self):
        self._element = None

    def start_object(self, object_name, object_id):
        self._element = et.Element(object_name, attrib={'id': object_id})

    def add_property(self, name, value):
        prop = et.SubElement(self._element, name)
        prop.text = value

    def to_str(self):
        return et.tostring(self._element, encoding='unicode')
```

**注意:**上面的例子没有实现一个完整的`Serializer`接口，但是对于我们的目的和演示工厂方法来说已经足够好了。

由于 [Python](https://www.python.org/) 语言的动态特性，`Serializer`接口是一个抽象的概念。像 [Java](https://realpython.com/oop-in-python-vs-java/) 或 C#这样的静态语言要求显式定义接口。在 Python 中，任何提供所需方法或函数的对象都被称为实现了接口。该示例将`Serializer`接口定义为实现以下方法或函数的对象:

*   `.start_object(object_name, object_id)`
*   `.add_property(name, value)`
*   `.to_str()`

这个接口是由具体的类`JsonSerializer`和`XmlSerializer`实现的。

最初的例子使用了一个`SongSerializer`类。对于新的应用程序，您将实现一些更通用的东西，比如`ObjectSerializer`:

```py
# In serializers.py

class ObjectSerializer:
    def serialize(self, serializable, format):
        serializer = factory.get_serializer(format)
        serializable.serialize(serializer)
        return serializer.to_str()
```

`ObjectSerializer`的实现是完全通用的，它只提到了一个`serializable`和一个`format`作为参数。

`format`用于标识`Serializer`的具体实现，由`factory`对象解析。`serializable`参数指的是另一个抽象接口，它应该在您想要序列化的任何对象类型上实现。

让我们来看看`Song`类中`serializable`接口的具体实现:

```py
# In songs.py

class Song:
    def __init__(self, song_id, title, artist):
        self.song_id = song_id
        self.title = title
        self.artist = artist

    def serialize(self, serializer):
        serializer.start_object('song', self.song_id)
        serializer.add_property('title', self.title)
        serializer.add_property('artist', self.artist)
```

`Song`类通过提供一个`.serialize(serializer)`方法来实现`Serializable`接口。在该方法中，`Song`类使用`serializer`对象来编写自己的信息，而不需要任何格式知识。

事实上，`Song`类甚至不知道目标是将数据转换成字符串。这很重要，因为您可以使用这个接口来提供不同种类的`serializer`，如果需要的话，它可以将`Song`信息转换成完全不同的表示。例如，您的应用程序将来可能需要将`Song`对象转换成二进制格式。

到目前为止，我们已经看到了客户端(`ObjectSerializer`)和产品(`serializer`)的实现。是时候完成工厂方法的实现并提供创建者了。例子中的创建者是`ObjectSerializer.serialize()`中的[变量](https://realpython.com/python-variables/) `factory`。

### 作为对象工厂的工厂方法

在最初的例子中，您将 creator 实现为一个函数。对于非常简单的例子来说，函数很好，但是当需求改变时，它们不能提供太多的灵活性。

类可以提供额外的接口来添加功能，并且可以派生它们来自定义行为。除非你有一个非常基本的将来永远不会改变的 creator，你想把它实现成一个类而不是一个函数。这些类型的类被称为对象工厂。

在`ObjectSerializer.serialize()`的实现中可以看到`SerializerFactory`的基本接口。该方法使用`factory.get_serializer(format)`从对象工厂中检索`serializer`。

您现在将实现`SerializerFactory`来满足这个接口:

```py
# In serializers.py

class SerializerFactory:
    def get_serializer(self, format):
        if format == 'JSON':
            return JsonSerializer()
        elif format == 'XML':
            return XmlSerializer()
        else:
            raise ValueError(format)

factory = SerializerFactory()
```

`.get_serializer()`的当前实现与您在原始示例中使用的相同。该方法评估`format`的值，并决定创建和返回的具体实现。这是一个相对简单的解决方案，允许我们验证所有工厂方法组件的功能。

让我们转到 Python 交互式解释器，看看它是如何工作的:

>>>

```py
>>> import songs
>>> import serializers
>>> song = songs.Song('1', 'Water of Love', 'Dire Straits')
>>> serializer = serializers.ObjectSerializer()

>>> serializer.serialize(song, 'JSON')
'{"id": "1", "title": "Water of Love", "artist": "Dire Straits"}'

>>> serializer.serialize(song, 'XML')
'<song id="1"><title>Water of Love</title><artist>Dire Straits</artist></song>'

>>> serializer.serialize(song, 'YAML')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "./serializers.py", line 39, in serialize
    serializer = factory.get_serializer(format)
  File "./serializers.py", line 52, in get_serializer
    raise ValueError(format)
ValueError: YAML
```

工厂方法的新设计允许应用程序通过添加新的类来引入新的特性，而不是改变现有的类。您可以通过在其他对象上实现`Serializable`接口来序列化它们。您可以通过在另一个类中实现`Serializer`接口来支持新格式。

缺少的部分是`SerializerFactory`必须改变以包括对新格式的支持。这个问题在新设计中很容易解决，因为`SerializerFactory`是一个类。

[*Remove ads*](/account/join/)

### 支持附加格式

当引入新格式时,`SerializerFactory`的当前实现需要改变。您的应用程序可能永远不需要支持任何额外的格式，但是您永远不知道。

您希望您的设计是灵活的，正如您将看到的，支持额外的格式而不改变`SerializerFactory`是相对容易的。

想法是在`SerializerFactory`中提供一个方法，为我们想要支持的格式注册一个新的`Serializer`实现:

```py
# In serializers.py

class SerializerFactory:

    def __init__(self):
        self._creators = {}

    def register_format(self, format, creator):
        self._creators[format] = creator

    def get_serializer(self, format):
        creator = self._creators.get(format)
        if not creator:
            raise ValueError(format)
        return creator()

factory = SerializerFactory()
factory.register_format('JSON', JsonSerializer)
factory.register_format('XML', XmlSerializer)
```

`.register_format(format, creator)`方法允许通过指定一个用于识别格式的`format`值和一个`creator`对象来注册新格式。creator 对象恰好是具体`Serializer`的类名。这是可能的，因为所有的`Serializer`类都提供了默认的`.__init__()`来初始化实例。

注册信息存储在`_creators` [字典](https://realpython.com/python-dicts/)中。`.get_serializer()`方法检索注册的创建者并创建所需的对象。如果所请求的`format`尚未注册，则`ValueError`被引发。

您现在可以通过实现一个`YamlSerializer`来验证设计的灵活性，并去掉您之前看到的烦人的`ValueError`:

```py
# In yaml_serializer.py

import yaml
import serializers

class YamlSerializer(serializers.JsonSerializer):
    def to_str(self):
        return yaml.dump(self._current_object)

serializers.factory.register_format('YAML', YamlSerializer)
```

**注意:**要实现这个例子，您需要使用`pip install PyYAML`在您的环境中安装 [`PyYAML`](https://pypi.org/project/PyYAML) 。

JSON 和 YAML 是非常相似的格式，所以你可以重用`JsonSerializer`的大部分实现，覆盖`.to_str()`来完成实现。然后用`factory`对象注册该格式，使其可用。

让我们使用 Python 交互式解释器来看看结果:

>>>

```py
>>> import serializers
>>> import songs
>>> import yaml_serializer
>>> song = songs.Song('1', 'Water of Love', 'Dire Straits')
>>> serializer = serializers.ObjectSerializer()

>>> print(serializer.serialize(song, 'JSON'))
{"id": "1", "title": "Water of Love", "artist": "Dire Straits"}

>>> print(serializer.serialize(song, 'XML'))
<song id="1"><title>Water of Love</title><artist>Dire Straits</artist></song>

>>> print(serializer.serialize(song, 'YAML'))
{artist: Dire Straits, id: '1', title: Water of Love}
```

通过使用对象工厂实现工厂方法并提供注册接口，您能够支持新的格式，而无需更改任何现有的应用程序代码。这将破坏现有功能或引入细微错误的风险降至最低。

## 通用对象工厂

`SerializerFactory`的实现是对原始示例的巨大改进。它提供了很大的灵活性来支持新的格式，并避免修改现有的代码。

尽管如此，当前的实现是专门针对上面的序列化问题的，它在其他上下文中不可重用。

工厂方法可以用来解决广泛的问题。当需求改变时，对象工厂为设计提供了额外的灵活性。理想情况下，您会想要一个无需复制实现就可以在任何情况下重用的对象工厂实现。

提供对象工厂的通用实现存在一些挑战，在接下来的部分中，您将关注这些挑战并实现一个可以在任何情况下重用的解决方案。

### 并非所有对象都可以被创建为相同的

实现通用对象工厂的最大挑战是，并非所有对象都是以相同的方式创建的。

并非所有情况都允许我们使用默认的`.__init__()`来创建和初始化对象。创建者(在本例中是对象工厂)返回完全初始化的对象是很重要的。

这一点很重要，因为如果不这样做，客户机就必须完成初始化，并使用复杂的条件代码来完全初始化所提供的对象。这违背了工厂方法设计模式的目的。

为了理解通用解决方案的复杂性，让我们看一个不同的问题。假设一个应用程序想要集成不同的音乐服务。这些服务可以在应用程序外部，也可以在应用程序内部，以便支持本地音乐收藏。每种服务都有不同的需求。

**注意:**我为这个例子定义的需求是为了说明的目的，并不反映你将不得不实现的与像 [Pandora](https://www.pandora.com) 或 [Spotify](https://www.spotify.com) 这样的服务集成的真实需求。

目的是提供一组不同的需求，展示实现通用对象工厂的挑战。

假设应用程序想要与 Spotify 提供的服务集成。该服务需要一个授权过程，在该过程中，提供客户端密钥和秘密用于授权。

该服务返回应该在任何进一步的通信中使用的访问代码。这个授权过程非常慢，而且应该只执行一次，所以应用程序希望保留初始化的服务对象，并在每次需要与 Spotify 通信时使用它。

与此同时，其他用户希望与 Pandora 集成。潘多拉可能会使用完全不同的授权过程。它还需要一个客户端密钥和秘密，但是它返回一个应该用于其他通信的消费者密钥和秘密。与 Spotify 一样，授权过程很慢，而且应该只执行一次。

最后，应用程序实现了本地音乐服务的概念，音乐集合存储在本地。该服务要求指定音乐集合在本地系统中的位置。创建新的服务实例非常快，因此每当用户想要访问音乐集合时，都可以创建一个新的实例。

这个例子提出了几个挑战。每个服务都用一组不同的参数初始化。此外，Spotify 和 Pandora 在创建服务实例之前需要一个授权过程。

他们还希望重用该实例，以避免多次授权应用程序。本地服务更简单，但它与其他服务的初始化接口不匹配。

在下面几节中，您将通过一般化创建接口和实现通用对象工厂来解决这个问题。

[*Remove ads*](/account/join/)

### 单独创建对象以提供公共接口

每个具体音乐服务的创建都有自己的一套要求。这意味着每个服务实现的公共初始化接口是不可能的，也不推荐这样做。

最好的方法是定义一种新类型的对象，它提供一个通用接口并负责创建一个具体的服务。这种新型物体将被称为`Builder`。`Builder`对象拥有创建和初始化服务实例的所有逻辑。您将为每个支持的服务实现一个`Builder`对象。

让我们先来看看应用程序配置:

```py
# In program.py

config = {
    'spotify_client_key': 'THE_SPOTIFY_CLIENT_KEY',
    'spotify_client_secret': 'THE_SPOTIFY_CLIENT_SECRET',
    'pandora_client_key': 'THE_PANDORA_CLIENT_KEY',
    'pandora_client_secret': 'THE_PANDORA_CLIENT_SECRET',
    'local_music_location': '/usr/data/music'
}
```

`config`字典包含初始化每个服务所需的所有值。下一步是定义一个接口，该接口将使用这些值来创建音乐服务的具体实现。该接口将在一个`Builder`中实现。

让我们看看`SpotifyService`和`SpotifyServiceBuilder`的实现:

```py
# In music.py

class SpotifyService:
    def __init__(self, access_code):
        self._access_code = access_code

    def test_connection(self):
        print(f'Accessing Spotify with {self._access_code}')

class SpotifyServiceBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, spotify_client_key, spotify_client_secret, **_ignored):
        if not self._instance:
            access_code = self.authorize(
                spotify_client_key, spotify_client_secret)
            self._instance = SpotifyService(access_code)
        return self._instance

    def authorize(self, key, secret):
        return 'SPOTIFY_ACCESS_CODE'
```

**注意:**音乐服务接口定义了一个`.test_connection()`方法，对于演示来说应该足够了。

这个例子展示了一个实现了`.__call__(spotify_client_key, spotify_client_secret, **_ignored)`的`SpotifyServiceBuilder`。

该方法用于创建和初始化具体的`SpotifyService`。它指定所需的参数，并忽略通过`**_ignored`提供的任何附加参数。一旦检索到`access_code`，它就创建并返回`SpotifyService`实例。

请注意，`SpotifyServiceBuilder`保留了服务实例，并且只在第一次请求服务时创建一个新实例。这避免了在需求中多次经历授权过程。

让我们为潘多拉做同样的事情:

```py
# In music.py

class PandoraService:
    def __init__(self, consumer_key, consumer_secret):
        self._key = consumer_key
        self._secret = consumer_secret

    def test_connection(self):
        print(f'Accessing Pandora with {self._key} and {self._secret}')

class PandoraServiceBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, pandora_client_key, pandora_client_secret, **_ignored):
        if not self._instance:
            consumer_key, consumer_secret = self.authorize(
                pandora_client_key, pandora_client_secret)
            self._instance = PandoraService(consumer_key, consumer_secret)
        return self._instance

    def authorize(self, key, secret):
        return 'PANDORA_CONSUMER_KEY', 'PANDORA_CONSUMER_SECRET'
```

`PandoraServiceBuilder`实现了相同的接口，但是它使用不同的参数和过程来创建和初始化`PandoraService`。它还保留了服务实例，因此授权只发生一次。

最后，让我们看看本地服务实现:

```py
# In music.py

class LocalService:
    def __init__(self, location):
        self._location = location

    def test_connection(self):
        print(f'Accessing Local music at {self._location}')

def create_local_music_service(local_music_location, **_ignored):
    return LocalService(local_music_location)
```

`LocalService`只需要一个存储集合的位置来初始化`LocalService`。

每次请求服务时都会创建一个新的实例，因为没有缓慢的授权过程。要求更简单，不需要`Builder`类。相反，使用返回初始化的`LocalService`的函数。该函数匹配构建器类中实现的`.__call__()`方法的接口。

### 对象工厂的通用接口

通用对象工厂(`ObjectFactory`)可以利用通用的`Builder`接口来创建各种对象。它提供了一个基于`key`值注册`Builder`的方法和一个基于`key`创建具体对象实例的方法。

让我们看看我们的泛型`ObjectFactory`的实现:

```py
# In object_factory.py

class ObjectFactory:
    def __init__(self):
        self._builders = {}

    def register_builder(self, key, builder):
        self._builders[key] = builder

    def create(self, key, **kwargs):
        builder = self._builders.get(key)
        if not builder:
            raise ValueError(key)
        return builder(**kwargs)
```

`ObjectFactory`的实现结构和你在`SerializerFactory`中看到的一样。

不同之处在于接口，该接口公开以支持创建任何类型的对象。构建器参数可以是实现[可调用](https://docs.python.org/3.7/library/functions.html#callable)接口的任何对象。这意味着`Builder`可以是实现`.__call__()`的函数、类或对象。

`.create()`方法要求额外的参数被指定为关键字参数。这允许`Builder`对象指定它们需要的参数，并忽略其余的参数。例如，您可以看到`create_local_music_service()`指定了一个`local_music_location`参数，并忽略了其余的参数。

让我们创建工厂实例，并为您想要支持的服务注册构建器:

```py
# In music.py
import object_factory

# Omitting other implementation classes shown above

factory = object_factory.ObjectFactory()
factory.register_builder('SPOTIFY', SpotifyServiceBuilder())
factory.register_builder('PANDORA', PandoraServiceBuilder())
factory.register_builder('LOCAL', create_local_music_service)
```

`music`模块通过`factory`属性公开`ObjectFactory`实例。然后，构建器向实例注册。对于 Spotify 和 Pandora，你注册了它们对应的 builder 的一个实例，但是对于本地服务，你只是传递了函数。

让我们编写一个演示该功能的小程序:

```py
# In program.py
import music

config = {
    'spotify_client_key': 'THE_SPOTIFY_CLIENT_KEY',
    'spotify_client_secret': 'THE_SPOTIFY_CLIENT_SECRET',
    'pandora_client_key': 'THE_PANDORA_CLIENT_KEY',
    'pandora_client_secret': 'THE_PANDORA_CLIENT_SECRET',
    'local_music_location': '/usr/data/music'
}

pandora = music.factory.create('PANDORA', **config)
pandora.test_connection()

spotify = music.factory.create('SPOTIFY', **config)
spotify.test_connection()

local = music.factory.create('LOCAL', **config)
local.test_connection()

pandora2 = music.services.get('PANDORA', **config)
print(f'id(pandora) == id(pandora2): {id(pandora) == id(pandora2)}')

spotify2 = music.services.get('SPOTIFY', **config)
print(f'id(spotify) == id(spotify2): {id(spotify) == id(spotify2)}')
```

应用程序定义了一个代表应用程序配置的`config`字典。配置被用作工厂的关键字参数，与您想要访问的服务无关。工厂根据指定的`key`参数创建音乐服务的具体实现。

您现在可以运行我们的程序来看看它是如何工作的:

```py
$ python program.py
Accessing Pandora with PANDORA_CONSUMER_KEY and PANDORA_CONSUMER_SECRET
Accessing Spotify with SPOTIFY_ACCESS_CODE
Accessing Local music at /usr/data/music
id(pandora) == id(pandora2): True
id(spotify) == id(spotify2): True
```

您可以看到根据指定的服务类型创建了正确的实例。您还可以看到，请求 Pandora 或 Spotify 服务总是返回相同的实例。

[*Remove ads*](/account/join/)

## 专门化对象工厂以提高代码可读性

通用解决方案是可重用的，并且避免了代码重复。不幸的是，它们也会模糊代码，降低可读性。

上例显示，要访问音乐服务，需要调用`music.factory.create()`。这可能会导致混乱。其他开发人员可能认为每次都会创建一个新的实例，并决定他们应该保留服务实例以避免缓慢的初始化过程。

您知道不会发生这种情况，因为`Builder`类保留了初始化的实例并返回它以供后续调用，但是从阅读代码来看这并不清楚。

一个好的解决方案是专门化一个通用的实现来提供一个特定于应用程序上下文的接口。在这一节中，您将在我们的音乐服务环境中专门化`ObjectFactory`,这样应用程序代码就能更好地传达意图，变得更具可读性。

以下示例显示了如何专门化`ObjectFactory`，为应用程序的上下文提供一个显式接口:

```py
# In music.py

class MusicServiceProvider(object_factory.ObjectFactory):
    def get(self, service_id, **kwargs):
        return self.create(service_id, **kwargs)

services = MusicServiceProvider()
services.register_builder('SPOTIFY', SpotifyServiceBuilder())
services.register_builder('PANDORA', PandoraServiceBuilder())
services.register_builder('LOCAL', create_local_music_service)
```

你从`ObjectFactory`中派生出`MusicServiceProvider`，并公开了一个新方法`.get(service_id, **kwargs)`。

这个方法调用泛型`.create(key, **kwargs)`，所以行为保持不变，但是代码在我们的应用程序上下文中读起来更好。您还将之前的`factory`变量重命名为`services`，并将其初始化为`MusicServiceProvider`。

如您所见，更新后的应用程序代码现在看起来更好了:

```py
import music

config = {
    'spotify_client_key': 'THE_SPOTIFY_CLIENT_KEY',
    'spotify_client_secret': 'THE_SPOTIFY_CLIENT_SECRET',
    'pandora_client_key': 'THE_PANDORA_CLIENT_KEY',
    'pandora_client_secret': 'THE_PANDORA_CLIENT_SECRET',
    'local_music_location': '/usr/data/music'
}

pandora = music.services.get('PANDORA', **config)
pandora.test_connection()
spotify = music.services.get('SPOTIFY', **config)
spotify.test_connection()
local = music.services.get('LOCAL', **config)
local.test_connection()

pandora2 = music.services.get('PANDORA', **config)
print(f'id(pandora) == id(pandora2): {id(pandora) == id(pandora2)}')

spotify2 = music.services.get('SPOTIFY', **config)
print(f'id(spotify) == id(spotify2): {id(spotify) == id(spotify2)}')
```

运行程序表明行为没有改变:

```py
$ python program.py
Accessing Pandora with PANDORA_CONSUMER_KEY and PANDORA_CONSUMER_SECRET
Accessing Spotify with SPOTIFY_ACCESS_CODE
Accessing Local music at /usr/data/music
id(pandora) == id(pandora2): True
id(spotify) == id(spotify2): True
```

## 结论

工厂方法是一种广泛使用的、创造性的设计模式，可以用在许多存在多个具体接口实现的情况下。

该模式删除了难以维护的复杂逻辑代码，并用可重用和可扩展的设计取而代之。该模式避免修改现有代码来支持新的需求。

这一点很重要，因为更改现有代码可能会引入行为变化或细微的错误。

在本文中，您了解了:

*   工厂方法设计模式是什么，它的组件是什么
*   如何重构现有代码以利用工厂方法
*   应该使用工厂方法的情况
*   对象工厂如何为实现工厂方法提供更大的灵活性
*   如何实现通用对象工厂及其挑战
*   如何专门化一个通用解决方案来提供一个更好的环境

## 延伸阅读

如果你想学习更多关于工厂方法和其他设计模式的知识，我推荐 GoF 的[Design Patterns:Elements of Reusable Object-Oriented Software](https://realpython.com/asins/0201633612/)，这是一个广泛采用的设计模式的很好的参考。

此外， [Heads First Design Patterns:一本由 Eric Freeman 和 Elisabeth Robson 编写的对大脑友好的指南](https://realpython.com/asins/0596007124/)提供了一个有趣、易读的设计模式解释。

维基百科有一个很好的[设计模式](https://en.wikipedia.org/wiki/Software_design_pattern)的目录，里面有最常见和最有用模式的链接。******