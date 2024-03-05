# Python 语音识别终极指南

> 原文：<https://realpython.com/python-speech-recognition/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和文字教程一起看，加深理解: [**用 Python 进行语音识别**](/courses/speech-recognition-python/)

您是否想过如何在您的 Python 项目中添加语音识别？如果有，那就继续读下去吧！这比你想象的要容易。

像亚马逊 Alexa 这样的语音功能产品的巨大成功已经证明，在可预见的未来，某种程度的语音支持将是家用技术的一个重要方面，这远远不是一种时尚。如果你仔细想想，原因是显而易见的。将语音识别集成到您的 Python 应用程序中，可以提供很少技术可以比拟的交互性和可访问性。

仅可访问性的改进就值得考虑。语音识别允许老年人、身体和视力受损者快速、自然地与最先进的产品和服务进行交互，无需 GUI！

最重要的是，在 Python 项目中包含语音识别非常简单。在本指南中，您将了解如何做到这一点。您将了解到:

*   语音识别的工作原理，
*   PyPI 上有哪些软件包；和
*   如何安装和使用 SpeechRecognition 包——一个功能齐全且易于使用的 Python 语音识别库。

最后，您将把您所学到的知识应用到一个简单的“猜单词”游戏中，看看它是如何组合在一起的。

**免费奖励:** [单击此处下载一个 Python 语音识别示例项目，该项目具有完整的源代码](#)，您可以将其用作自己的语音识别应用程序的基础。

## 语音识别的工作原理——概述

在我们开始用 Python 进行语音识别的基础知识之前，让我们花点时间来讨论一下语音识别是如何工作的。完整的讨论可以写一本书，所以我不会在这里用所有的技术细节来烦你。事实上，这一部分并不是本教程其余部分的先决条件。如果你想直奔主题，那就直接跳到前面。

语音识别源于 20 世纪 50 年代早期贝尔实验室的研究。早期的系统仅限于单个说话者，并且词汇量有限，大约只有十几个单词。现代语音识别系统与古代的系统相比已经有了很大的进步。它们可以识别来自多个说话者的语音，并拥有多种语言的大量词汇。

语音识别的第一个组成部分当然是语音。语音必须通过麦克风从物理声音转换为电信号，然后通过模数转换器转换为数字数据。一旦数字化，几个模型可以用来转录音频到文本。

大多数现代语音识别系统依赖于所谓的[隐马尔可夫模型](https://en.wikipedia.org/wiki/Hidden_Markov_model) (HMM)。这种方法基于这样一种假设，即当在足够短的时间尺度(比如 10 毫秒)上观察时，语音信号可以合理地近似为一个平稳过程，即统计特性不随时间变化的过程。

在典型的 HMM 中，语音信号被分成 10 毫秒的片段。每个碎片的功率谱本质上是信号功率与频率的函数关系图，它被映射到一个称为[倒谱](https://en.wikipedia.org/wiki/Cepstrum)系数的实数向量上。这个向量的维数通常很小，有时低至 10，尽管更精确的系统可能有 32 或更高的维数。HMM 的最终输出是这些向量的序列。

为了将语音解码成文本，向量组与一个或多个[音素](https://en.wikipedia.org/wiki/Phoneme)匹配——语音的基本单位。这种计算需要训练，因为音素的声音因说话者而异，甚至因同一说话者的不同话语而异。然后应用一种特殊的算法来确定产生给定音素序列的最可能的单词。

可以想象，这整个过程在计算上可能是昂贵的。在许多现代语音识别系统中，神经网络用于简化语音信号，在 HMM 识别之前使用特征变换和维度缩减技术*。语音活动检测器(VAD)也用于将音频信号减少到可能包含语音的部分。这防止识别器浪费时间分析信号的不必要部分。*

幸运的是，作为一名 Python 程序员，您不必担心这些。许多语音识别服务可以通过 API 在线使用，其中许多服务提供了[Python SDK](https://realpython.com/api-integration-in-python/)。

[*Remove ads*](/account/join/)

## 挑选一个 Python 语音识别包

PyPI 上有一些用于语音识别的包。其中一些包括:

*   [猴子](https://pypi.org/project/apiai/)
*   [assemblyai](https://pypi.org/project/assemblyai/)
*   [谷歌云语音](https://pypi.org/project/google-cloud-speech/)
*   [pocketsphinx](https://pypi.org/project/pocketsphinx/)
*   [演讲人识别](https://pypi.org/project/SpeechRecognition/)
*   [沃森-开发者-云](https://pypi.org/project/watson-developer-cloud/)
*   [机智](https://pypi.org/project/wit/)

其中一些软件包——如 wit 和 apai——提供内置功能，如用于识别说话者意图的[自然语言处理](https://realpython.com/nltk-nlp-python/),这超出了基本的语音识别。其他的，比如 google-cloud-speech，只专注于语音到文本的转换。

有一个软件包在易用性方面非常突出:SpeechRecognition。

识别语音需要音频输入，而 SpeechRecognition 使检索这种输入变得非常容易。SpeechRecognition 无需从头开始编写访问麦克风和处理音频文件的脚本，只需几分钟就能让您启动并运行。

SpeechRecognition 库充当几种流行的语音 API 的包装器，因此非常灵活。其中之一——Google Web Speech API——支持一个默认的 API 键，该键被硬编码到 SpeechRecognition 库中。这意味着你可以不用注册服务就可以开始工作。

SpeechRecognition 包的灵活性和易用性使其成为任何 Python 项目的绝佳选择。然而，并不能保证支持它所包装的每个 API 的每个特性。您需要花一些时间研究可用的选项，看看演讲识别功能是否适用于您的特定情况。

所以，既然您已经确信应该试用 SpeechRecognition，下一步就是在您的环境中安装它。

## 安装语音识别功能

SpeechRecognition 与 Python 2.6、2.7 和 3.3+兼容，但是对于 Python 2 需要一些[额外的安装步骤。对于本教程，我假设您使用的是 Python 3.3+。](https://github.com/Uberi/speech_recognition#requirements)

您可以使用 pip 从终端安装 SpeechRecognition:

```py
$ pip install SpeechRecognition
```

安装后，您应该通过打开解释器会话并键入以下命令来验证安装:

>>>

```py
>>> import speech_recognition as sr
>>> sr.__version__
'3.8.1'
```

**注意:**您获得的版本号可能会有所不同。在撰写本文时，最新的版本是 3.8.1。

继续并保持此会话打开。你很快就会开始使用它。

如果你需要做的只是处理现有的音频文件，那么演讲识别*将*开箱即用。然而，特定的用例需要一些依赖关系。值得注意的是，捕捉麦克风输入需要 PyAudio 包。

随着阅读的深入，您将会看到您需要哪些依赖项。现在，让我们深入研究一下这个包的基础知识。

## `Recognizer`类

SpeechRecognition 的所有神奇之处都发生在`Recognizer`类上。

当然，`Recognizer`实例的主要目的是识别语音。每个实例都带有各种设置和功能，用于识别来自音频源的语音。

创建一个`Recognizer`实例很容易。在当前的解释器会话中，只需键入:

>>>

```py
>>> r = sr.Recognizer()
```

每个`Recognizer`实例都有七种方法，用于使用各种 API 识别来自音频源的语音。这些是:

*   `recognize_bing()` : [微软必应演讲](https://azure.microsoft.com/en-us/services/cognitive-services/speech/)
*   `recognize_google()` : [谷歌网络语音 API](https://w3c.github.io/speech-api/speechapi.html)
*   `recognize_google_cloud()` : [谷歌云语音](https://cloud.google.com/speech/)——需要安装谷歌云语音包
*   `recognize_houndify()` : [用 SoundHound 来形容](https://www.houndify.com/)
*   `recognize_ibm()` : [IBM 语音转文字](https://www.ibm.com/watson/services/speech-to-text/)
*   `recognize_sphinx()` : [CMU 狮身人面像](https://cmusphinx.github.io/) -需要安装 PocketSphinx
*   `recognize_wit()` : [Wit.ai](https://wit.ai/)

在这七个引擎中，只有`recognize_sphinx()`离线使用 CMU 狮身人面像引擎。其他六个都需要互联网连接。

对每个 API 的特性和优点的全面讨论超出了本教程的范围。因为 SpeechRecognition 为 Google Web Speech API 提供了一个默认的 API 键，所以您可以马上开始使用它。因此，我们将在本指南中使用 Web 语音 API。其他六个 API 都需要使用 API 密钥或用户名/密码组合进行身份验证。欲了解更多信息，请查阅演讲人认知[文档](https://github.com/Uberi/speech_recognition/blob/master/reference/library-reference.rst)。

**注意:**speech recognition 提供的默认密钥仅用于测试目的，**谷歌可能随时撤销该密钥**。在产品中使用谷歌网络语音 API 不是一个好主意*。即使有一个有效的 API 密匙，你也将被限制在每天只有 50 个请求，并且[没有办法提高这个配额](http://www.chromium.org/developers/how-tos/api-keys)。幸运的是，SpeechRecognition 的界面对于每个 API 几乎都是相同的，因此您今天所学的内容将很容易转化为现实世界的项目。*

如果 API 不可达，每个`recognize_*()`方法将抛出一个`speech_recognition.RequestError`异常。对于`recognize_sphinx()`来说，这可能是由于 Sphinx 安装缺失、损坏或不兼容造成的。对于其他六种方法，如果达到配额限制、服务器不可用或没有互联网连接，可能会抛出`RequestError`。

好了，闲聊够了。让我们把手弄脏吧。继续尝试在您的解释器会话中调用`recognize_google()`。

>>>

```py
>>> r.recognize_google()
```

发生了什么事？

您可能会看到类似这样的内容:

```py
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: recognize_google() missing 1 required positional argument: 'audio_data'
```

你可能已经猜到会发生这种情况。从无到有怎么可能认识到什么？

`Recognizer`类的所有七个`recognize_*()`方法都需要一个`audio_data`参数。在每种情况下，`audio_data`必须是 SpeechRecognition 的`AudioData`类的一个实例。

创建`AudioData`实例有两种方法:从音频文件或麦克风录制的音频。音频文件比较容易上手，所以我们先来看看。

[*Remove ads*](/account/join/)

## 使用音频文件

继续之前，您需要下载一个音频文件。我用来入门的那个“harvard.wav”，可以在这里找到[。确保将它保存到运行 Python 解释器会话的同一目录中。](https://github.com/realpython/python-speech-recognition)

SpeechRecognition 通过其便利的`AudioFile`类使处理音频文件变得简单。这个类可以用音频文件的路径初始化，并提供一个上下文管理器接口来读取和处理文件的内容。

### 支持的文件类型

目前，SpeechRecognition 支持以下文件格式:

*   WAV:必须是 PCM/LPCM 格式
*   AIFF
*   AIFF-C
*   FLAC:必须是原生 FLAC 格式；不支持 OGG-FLAC

如果你在基于 x-86 的 Linux、macOS 或 Windows 上工作，你应该能够毫无问题地处理 FLAC 文件。在其他平台上，您需要安装 FLAC 编码器，并确保您可以访问`flac`命令行工具。如果适用于您，您可以在这里找到更多信息[。](https://xiph.org/flac/)

### 使用`record()`从文件中捕获数据

在您的解释器会话中键入以下内容，以处理“harvard.wav”文件的内容:

>>>

```py
>>> harvard = sr.AudioFile('harvard.wav')
>>> with harvard as source:
...    audio = r.record(source)
...
```

上下文管理器打开文件并读取其内容，将数据存储在名为`source.`的`AudioFile`实例中，然后`record()`方法将整个文件中的数据记录到一个`AudioData`实例中。您可以通过检查`audio`的类型来确认这一点:

>>>

```py
>>> type(audio)
<class 'speech_recognition.AudioData'>
```

您现在可以调用`recognize_google()`来尝试识别音频中的任何语音。根据您的互联网连接速度，您可能需要等待几秒钟才能看到结果。

>>>

```py
>>> r.recognize_google(audio)
'the stale smell of old beer lingers it takes heat
to bring out the odor a cold dip restores health and
zest a salt pickle taste fine with ham tacos al
Pastore are my favorite a zestful food is the hot
cross bun'
```

恭喜你！您刚刚转录了您的第一个音频文件！

如果你想知道“harvard.wav”文件中的短语来自哪里，它们是哈佛句子的例子。这些短语由 IEEE 于 1965 年发布，用于电话线的语音清晰度测试。今天，它们仍然用于 VoIP 和蜂窝测试。

哈佛的句子由 72 个 10 个短语组成。你可以在 [Open Speech Repository](http://www.voiptroubleshooter.com/open_speech/index.html) 网站上找到这些短语的免费录音。录音有英语、汉语普通话、法语和印地语版本。它们为测试您的代码提供了极好的免费资料来源。

### 用`offset`和`duration` 捕捉片段

如果您只想在文件中捕获一部分语音，该怎么办？`record()`方法接受一个`duration`关键字参数，该参数在指定的秒数后停止记录。

例如，以下代码捕获文件前四秒钟的任何语音:

>>>

```py
>>> with harvard as source:
...     audio = r.record(source, duration=4)
...
>>> r.recognize_google(audio)
'the stale smell of old beer lingers'
```

在`with`块中使用的`record()`方法总是在文件流中向前移动。这意味着，如果您录制一次四秒钟，然后再次录制四秒钟，第二次将返回前四秒钟之后的四秒钟音频*。*

>>>

```py
>>> with harvard as source:
...     audio1 = r.record(source, duration=4)
...     audio2 = r.record(source, duration=4)
...
>>> r.recognize_google(audio1)
'the stale smell of old beer lingers'
>>> r.recognize_google(audio2)
'it takes heat to bring out the odor a cold dip'
```

注意`audio2`包含了文件中第三个短语的一部分。当指定持续时间时，录音可能会在短语中间停止，甚至在单词中间停止，这可能会影响转录的准确性。稍后会有更多的介绍。

除了指定记录持续时间之外，还可以使用`offset`关键字参数为`record()`方法指定一个特定的起点。该值表示从文件开始到开始记录之前要忽略的秒数。

要仅捕获文件中的第二个短语，您可以从 4 秒钟的偏移开始记录，比如说，3 秒钟。

>>>

```py
>>> with harvard as source:
...     audio = r.record(source, offset=4, duration=3)
...
>>> r.recognize_google(audio)
'it takes heat to bring out the odor'
```

如果您事先知道文件中的语音结构，那么`offset`和`duration`关键字参数对于分割音频文件*非常有用。然而，匆忙使用它们会导致糟糕的转录。要查看这种效果，请在您的解释器中尝试以下操作:*

>>>

```py
>>> with harvard as source:
...     audio = r.record(source, offset=4.7, duration=2.8)
...
>>> r.recognize_google(audio)
'Mesquite to bring out the odor Aiko'
```

通过在 4.7 秒开始记录，您错过了短语“需要加热才能带出气味”开头的“it t”部分，因此 API 只得到“akes heat”，它与“Mesquite”匹配。

同样，在录音的结尾，你听到了“a co”，这是第三个短语“冷浸恢复健康和热情”的开头这与 API 中的“Aiko”相匹配。

还有一个原因，你可能会得到不准确的转录。噪音！上面的例子工作得很好，因为音频文件相当干净。在现实世界中，除非您有机会事先处理音频文件，否则您不能期望音频是无噪声的。

[*Remove ads*](/account/join/)

### 噪声对语音识别的影响

噪音是生活的现实。所有的录音都有一定程度的噪音，未经处理的噪音会破坏语音识别应用的准确性。

要了解噪声如何影响语音识别，请在此下载“jackhammer.wav”文件[。像往常一样，确保将它保存到解释器会话的工作目录中。](https://github.com/realpython/python-speech-recognition)

这份文件有一句话“旧啤酒的陈腐气味挥之不去”，背景是一个响亮的手提钻。

当你试图转录这个文件时会发生什么？

>>>

```py
>>> jackhammer = sr.AudioFile('jackhammer.wav')
>>> with jackhammer as source:
...     audio = r.record(source)
...
>>> r.recognize_google(audio)
'the snail smell of old gear vendors'
```

太离谱了。

那么你如何处理这个问题呢？您可以尝试使用`Recognizer`类的`adjust_for_ambient_noise()`方法。

>>>

```py
>>> with jackhammer as source:
...     r.adjust_for_ambient_noise(source)
...     audio = r.record(source)
...
>>> r.recognize_google(audio)
'still smell of old beer vendors'
```

这让你离真正的短语更近了一点，但它仍然不完美。此外，短语开头缺少“the”。这是为什么呢？

`adjust_for_ambient_noise()`方法读取文件流的第一秒，并将识别器校准到音频的噪声级别。因此，在您调用`record()`来捕获数据之前，流的这一部分就被消耗掉了。

您可以使用`duration`关键字参数调整`adjust_for_ambient_noise()`用于分析的时间范围。该参数采用以秒为单位的数值，默认情况下设置为 1。尝试将该值降低到 0.5。

>>>

```py
>>> with jackhammer as source:
...     r.adjust_for_ambient_noise(source, duration=0.5)
...     audio = r.record(source)
...
>>> r.recognize_google(audio)
'the snail smell like old Beer Mongers'
```

好吧，那让你在短语的开始有了“the ”,但是现在你有一些新的问题了！有时不可能消除噪声的影响——信号噪声太大，无法成功处理。这份档案就是这样。

如果您发现自己经常遇到这些问题，您可能需要对音频进行一些预处理。这可以通过[音频编辑软件](https://www.audacityteam.org/)或者可以对文件应用过滤器的 Python 包(比如 [SciPy](https://realpython.com/python-scipy-cluster-optimize/) )来完成。关于这一点的详细讨论超出了本教程的范围——如果你感兴趣，可以看看艾伦·唐尼的 [Think DSP](http://greenteapress.com/wp/think-dsp/) 一书。现在，请注意，音频文件中的环境噪声可能会导致问题，为了最大限度地提高语音识别的准确性，必须解决这个问题。

当处理有噪声的文件时，查看实际的 API 响应会很有帮助。大多数 API 返回一个包含许多可能转写的 [JSON 字符串](https://realpython.com/python-json/)。`recognize_google()`方法将总是返回*最有可能的*转录，除非你强迫它给你完整的响应。

您可以通过将`recognize_google()`方法的`show_all`关键字参数设置为`True.`来做到这一点

>>>

```py
>>> r.recognize_google(audio, show_all=True)
{'alternative': [
 {'transcript': 'the snail smell like old Beer Mongers'}, 
 {'transcript': 'the still smell of old beer vendors'}, 
 {'transcript': 'the snail smell like old beer vendors'},
 {'transcript': 'the stale smell of old beer vendors'}, 
 {'transcript': 'the snail smell like old beermongers'}, 
 {'transcript': 'destihl smell of old beer vendors'}, 
 {'transcript': 'the still smell like old beer vendors'}, 
 {'transcript': 'bastille smell of old beer vendors'}, 
 {'transcript': 'the still smell like old beermongers'}, 
 {'transcript': 'the still smell of old beer venders'}, 
 {'transcript': 'the still smelling old beer vendors'}, 
 {'transcript': 'musty smell of old beer vendors'}, 
 {'transcript': 'the still smell of old beer vendor'}
], 'final': True}
```

如您所见，`recognize_google()`返回一个带有键`'alternative'`的字典，指向一个可能的抄本列表。该响应的结构可能因 API 而异，主要用于调试。

到目前为止，您已经对演讲识别包的基本知识有了很好的了解。您已经看到了如何从一个音频文件创建一个`AudioFile`实例，并使用`record()`方法从文件中捕获数据。您学习了如何使用`record()`的`offset`和`duration`关键字参数来记录文件片段，并且体验了噪声对转录准确性的不利影响。

现在是有趣的部分。让我们从转录静态音频文件过渡到通过接受麦克风输入来使您的项目具有交互性。

[*Remove ads*](/account/join/)

## 使用麦克风

要使用 SpeechRecognizer 访问您的麦克风，您必须安装 [PyAudio 软件包](https://people.csail.mit.edu/hubert/pyaudio/)。继续并关闭您当前的解释器会话，让我们这样做。

### 安装 PyAudio

安装 PyAudio 的过程会因您的操作系统而异。

#### Debian Linux

如果你在基于 Debian 的 Linux 上(比如 Ubuntu)，你可以用`apt`安装 PyAudio:

```py
$ sudo apt-get install python-pyaudio python3-pyaudio
```

一旦安装完毕，你可能仍然需要运行`pip install pyaudio`，尤其是如果你正在一个[虚拟环境](https://realpython.com/python-virtual-environments-a-primer/)中工作。

#### 苹果电脑

对于 macOS 来说，首先你需要用 Homebrew 安装 PortAudio，然后用 [`pip`](https://realpython.com/what-is-pip/) 安装 PyAudio:

```py
$ brew install portaudio
$ pip install pyaudio
```

#### 窗户

在 Windows 上，可以用 [`pip`](https://realpython.com/what-is-pip/) 安装 PyAudio:

```py
$ pip install pyaudio
```

#### 测试安装

一旦安装了 PyAudio，就可以从控制台测试安装。

```py
$ python -m speech_recognition
```

请确定您的默认麦克风已打开且未静音。如果安装工作正常，您应该会看到如下内容:

```py
A moment of silence, please...
Set minimum energy threshold to 600.4452854381937
Say something!
```

Go ahead and play around with it a little bit by speaking into your microphone and seeing how well SpeechRecognition transcribes your speech.

**注意:**如果你在 Ubuntu 上，得到一些像“ALSA lib …未知 PCM”这样的时髦输出，参考[这一页](https://github.com/Uberi/speech_recognition#on-ubuntudebian-i-get-annoying-output-in-the-terminal-saying-things-like-bt_audio_service_open--connection-refused-and-various-others)关于抑制这些信息的提示。这个输出来自与 Ubuntu 一起安装的 ALSA 软件包——而不是 SpeechRecognition 或 PyAudio。实际上，这些消息可能表明您的 ALSA 配置有问题，但是根据我的经验，它们不会影响您代码的功能。他们大多令人讨厌。

[*Remove ads*](/account/join/)

### `Microphone`类

打开另一个解释器会话并创建识别器类的一个实例。

>>>

```py
>>> import speech_recognition as sr
>>> r = sr.Recognizer()
```

现在，您将使用默认的系统麦克风，而不是使用音频文件作为源。您可以通过创建一个`Microphone`类的实例来访问它。

>>>

```py
>>> mic = sr.Microphone()
```

如果您的系统没有默认麦克风(例如在 [Raspberry Pi](https://realpython.com/python-raspberry-pi/) 上)，或者您想要使用默认麦克风之外的麦克风，您将需要通过提供设备索引来指定使用哪个麦克风。您可以通过调用`Microphone`类的`list_microphone_names()`静态方法来获得麦克风名称的列表。

>>>

```py
>>> sr.Microphone.list_microphone_names()
['HDA Intel PCH: ALC272 Analog (hw:0,0)',
 'HDA Intel PCH: HDMI 0 (hw:0,3)',
 'sysdefault',
 'front',
 'surround40',
 'surround51',
 'surround71',
 'hdmi',
 'pulse',
 'dmix', 
 'default']
```

请注意，您的输出可能与上面的示例不同。

麦克风的设备索引是其名称在由`list_microphone_names().`返回的列表中的索引。例如，给定上面的输出，如果您想要使用名为“front”的麦克风，它在列表中的索引为 3，您将创建一个麦克风实例，如下所示:

>>>

```py
>>> # This is just an example; do not run
>>> mic = sr.Microphone(device_index=3)
```

不过，对于大多数项目，您可能会想要使用默认的系统麦克风。

### 使用`listen()`捕捉麦克风输入

既然您已经准备好了一个`Microphone`实例，那么是时候捕获一些输入了。

就像`AudioFile`类一样，`Microphone`是一个上下文管理器。您可以使用`with`块中的`Recognizer`类的`listen()`方法来捕获来自麦克风的输入。该方法将音频源作为其第一个参数，并记录来自该源的输入，直到检测到无声。

>>>

```py
>>> with mic as source:
...     audio = r.listen(source)
...
```

一旦你执行了`with`块，试着对着你的麦克风说“hello”。稍等片刻，让解释器提示符再次显示。一旦"> > >"提示返回，您就可以识别语音了。

>>>

```py
>>> r.recognize_google(audio)
'hello'
```

如果提示音不再出现，您的麦克风很可能拾取了太多的环境噪音。您可以使用 `Ctrl` + `C` 来中断该过程，以获得您的提示。

要处理环境噪声，您需要使用`Recognizer`类的`adjust_for_ambient_noise()`方法，就像您在尝试理解嘈杂的音频文件时所做的那样。由于来自麦克风的输入远不如来自音频文件的输入可预测，因此在您收听麦克风输入时，最好这样做。

>>>

```py
>>> with mic as source:
...     r.adjust_for_ambient_noise(source)
...     audio = r.listen(source)
...
```

运行上述代码后，等待一秒钟让`adjust_for_ambient_noise()`完成它的工作，然后试着对着麦克风说“hello”。同样，在尝试识别语音之前，您必须等待解释器提示返回。

回想一下`adjust_for_ambient_noise()`分析音源一秒钟。如果这对你来说太长了，你可以用关键字参数`duration`来调整它。

演讲者识别文档建议使用不少于 0.5 秒的持续时间。在某些情况下，您可能会发现持续时间长于默认值一秒会产生更好的结果。您需要的最小值取决于麦克风的周围环境。不幸的是，这些信息在开发过程中通常是未知的。根据我的经验，一秒钟的默认持续时间对于大多数应用程序来说已经足够了。

[*Remove ads*](/account/join/)

### 处理无法识别的语音

尝试在 interpeter 中键入前面的代码示例，并在麦克风中制造一些难以理解的噪音。您应该得到类似这样的响应:

```py
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/david/real_python/speech_recognition_primer/venv/lib/python3.5/site-packages/speech_recognition/__init__.py", line 858, in recognize_google
    if not isinstance(actual_result, dict) or len(actual_result.get("alternative", [])) == 0: raise UnknownValueError()
speech_recognition.UnknownValueError
```

API 无法匹配文本的音频会引发一个`UnknownValueError`异常。你应该总是用 [`try`和`except`块包装对 API 的调用来处理这个异常](https://realpython.com/python-exceptions/)。

**注意**:要抛出异常，你可能需要付出比预期更多的努力。API 非常努力地转录任何声音。对我来说，即使是简短的咕哝声也会被翻译成“怎么样”这样的词。咳嗽、拍手和咂嘴都会引发异常。

## 综合起来:一个“猜单词”游戏

现在，您已经看到了使用 SpeechRecognition 软件包识别语音的基本知识，让我们将您新学到的知识用于编写一个小游戏，从列表中随机选择一个单词，并给用户三次猜测该单词的机会。

以下是完整的脚本:

```py
import random
import time

import speech_recognition as sr

def recognize_speech_from_mic(recognizer, microphone):
    """Transcribe speech from recorded from `microphone`.

 Returns a dictionary with three keys:
 "success": a boolean indicating whether or not the API request was
 successful
 "error":   `None` if no error occured, otherwise a string containing
 an error message if the API could not be reached or
 speech was unrecognizable
 "transcription": `None` if speech could not be transcribed,
 otherwise a string containing the transcribed text
 """
    # check that recognizer and microphone arguments are appropriate type
    if not isinstance(recognizer, sr.Recognizer):
        raise TypeError("`recognizer` must be `Recognizer` instance")

    if not isinstance(microphone, sr.Microphone):
        raise TypeError("`microphone` must be `Microphone` instance")

    # adjust the recognizer sensitivity to ambient noise and record audio
    # from the microphone
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    # set up the response object
    response = {
        "success": True,
        "error": None,
        "transcription": None
    }

    # try recognizing the speech in the recording
    # if a RequestError or UnknownValueError exception is caught,
    #     update the response object accordingly
    try:
        response["transcription"] = recognizer.recognize_google(audio)
    except sr.RequestError:
        # API was unreachable or unresponsive
        response["success"] = False
        response["error"] = "API unavailable"
    except sr.UnknownValueError:
        # speech was unintelligible
        response["error"] = "Unable to recognize speech"

    return response

if __name__ == "__main__":
    # set the list of words, maxnumber of guesses, and prompt limit
    WORDS = ["apple", "banana", "grape", "orange", "mango", "lemon"]
    NUM_GUESSES = 3
    PROMPT_LIMIT = 5

    # create recognizer and mic instances
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    # get a random word from the list
    word = random.choice(WORDS)

    # format the instructions string
    instructions = (
        "I'm thinking of one of these words:\n"
        "{words}\n"
        "You have {n} tries to guess which one.\n"
    ).format(words=', '.join(WORDS), n=NUM_GUESSES)

    # show instructions and wait 3 seconds before starting the game
    print(instructions)
    time.sleep(3)

    for i in range(NUM_GUESSES):
        # get the guess from the user
        # if a transcription is returned, break out of the loop and
        #     continue
        # if no transcription returned and API request failed, break
        #     loop and continue
        # if API request succeeded but no transcription was returned,
        #     re-prompt the user to say their guess again. Do this up
        #     to PROMPT_LIMIT times
        for j in range(PROMPT_LIMIT):
            print('Guess {}. Speak!'.format(i+1))
            guess = recognize_speech_from_mic(recognizer, microphone)
            if guess["transcription"]:
                break
            if not guess["success"]:
                break
            print("I didn't catch that. What did you say?\n")

        # if there was an error, stop the game
        if guess["error"]:
            print("ERROR: {}".format(guess["error"]))
            break

        # show the user the transcription
        print("You said: {}".format(guess["transcription"]))

        # determine if guess is correct and if any attempts remain
        guess_is_correct = guess["transcription"].lower() == word.lower()
        user_has_more_attempts = i < NUM_GUESSES - 1

        # determine if the user has won the game
        # if not, repeat the loop if user has more attempts
        # if no attempts left, the user loses the game
        if guess_is_correct:
            print("Correct! You win!".format(word))
            break
        elif user_has_more_attempts:
            print("Incorrect. Try again.\n")
        else:
            print("Sorry, you lose!\nI was thinking of '{}'.".format(word))
            break
```

让我们稍微分解一下。

`recognize_speech_from_mic()`函数将`Recognizer`和`Microphone`实例作为参数，并返回一个包含三个键的字典。第一个关键字`"success"`是一个布尔值，它指示 API 请求是否成功。第二个键`"error"`，或者是 [`None`](https://realpython.com/null-in-python/) ，或者是指示 API 不可用或者语音不可理解的错误消息。最后，`"transcription"`键包含麦克风录制的音频的转录。

该函数首先检查`recognizer`和`microphone`参数的类型是否正确，如果其中一个无效，则引发一个`TypeError`:

```py
if not isinstance(recognizer, sr.Recognizer):
    raise TypeError('`recognizer` must be `Recognizer` instance')

if not isinstance(microphone, sr.Microphone):
    raise TypeError('`microphone` must be a `Microphone` instance')
```

然后使用`listen()`方法记录麦克风输入:

```py
with microphone as source:
    recognizer.adjust_for_ambient_noise(source)
    audio = recognizer.listen(source)
```

每次调用`recognize_speech_from_mic()`函数时，`adjust_for_ambient_noise()`方法用于校准识别器以改变噪声条件。

接下来，`recognize_google()`被调用来转录录音中的任何讲话。一个`try...except`块用于捕捉`RequestError`和`UnknownValueError`异常并相应地处理它们。API 请求的成功、任何错误消息和转录的语音都存储在`response`字典的`success`、`error`和`transcription`键中，由`recognize_speech_from_mic()`函数返回。

```py
response = {
    "success": True,
    "error": None,
    "transcription": None
}

try:
    response["transcription"] = recognizer.recognize_google(audio)
except sr.RequestError:
    # API was unreachable or unresponsive
    response["success"] = False
    response["error"] = "API unavailable"
except sr.UnknownValueError:
    # speech was unintelligible
    response["error"] = "Unable to recognize speech"

return response
```

您可以通过将上述脚本保存到一个名为“guessing_game.py”的文件中并在解释器会话中运行以下命令来测试`recognize_speech_from_mic()`函数:

>>>

```py
>>> import speech_recognition as sr
>>> from guessing_game import recognize_speech_from_mic
>>> r = sr.Recognizer()
>>> m = sr.Microphone()
>>> recognize_speech_from_mic(r, m)
{'success': True, 'error': None, 'transcription': 'hello'}
>>> # Your output will vary depending on what you say
```

游戏本身非常简单。首先，声明单词列表、允许猜测的最大数量和提示限制:

```py
WORDS = ['apple', 'banana', 'grape', 'orange', 'mango', 'lemon']
NUM_GUESSES = 3
PROMPT_LIMIT = 5
```

接下来，创建一个`Recognizer`和`Microphone`实例，并从`WORDS`中选择一个随机单词:

```py
recognizer = sr.Recognizer()
microphone = sr.Microphone()
word = random.choice(WORDS)
```

在打印一些指令并等待 3 秒钟后，使用一个 [`for`循环](https://realpython.com/python-for-loop/)来管理每个用户猜测所选单词的尝试。在`for`循环中的第一件事是另一个`for`循环，它最多提示用户`PROMPT_LIMIT`次猜测，每次都试图用`recognize_speech_from_mic()`函数识别输入，并将返回的字典存储到本地变量`guess`。

如果`guess`的`"transcription"`键不是`None`，则用户的语音被转录，内部循环以`break`结束。如果语音未被转录并且`"success"`键被设置为`False`，则出现 API 错误，并且循环再次以`break`终止。否则，API 请求成功，但语音无法识别。警告用户并重复`for`循环，给用户当前尝试的另一次机会。

```py
for j in range(PROMPT_LIMIT):
    print('Guess {}. Speak!'.format(i+1))
    guess = recognize_speech_from_mic(recognizer, microphone)
    if guess["transcription"]:
        break
    if not guess["success"]:
        break
    print("I didn't catch that. What did you say?\n")
```

一旦内部`for`循环终止，就会检查`guess`字典中的错误。如果发生任何错误，显示错误信息，用`break`终止外部`for`循环，这将结束程序执行。

```py
if guess['error']:
    print("ERROR: {}".format(guess["error"]))
    break
```

如果没有任何错误，转录将与随机选择的单词进行比较。string 对象的`lower()`方法用于确保猜测与所选单词的更好匹配。API 可以将与单词“apple”匹配的语音返回为“Apple”*或*“Apple”，并且任何一个响应都应该算作正确答案。

如果猜测正确，用户获胜，游戏终止。如果用户是不正确的，并且有任何剩余的尝试，外部的`for`循环重复，并且检索新的猜测。否则，用户输掉游戏。

```py
guess_is_correct = guess["transcription"].lower() == word.lower()
user_has_more_attempts = i < NUM_GUESSES - 1

if guess_is_correct:
    print('Correct! You win!'.format(word))
    break
elif user_has_more_attempts:
    print('Incorrect. Try again.\n')
else:
    print("Sorry, you lose!\nI was thinking of '{}'.".format(word))
    break
```

运行时，输出如下所示:

```py
I'm thinking of one of these words:
apple, banana, grape, orange, mango, lemon
You have 3 tries to guess which one.

Guess 1\. Speak!
You said: banana
Incorrect. Try again.

Guess 2\. Speak!
You said: lemon
Incorrect. Try again.

Guess 3\. Speak!
You said: Orange
Correct! You win!
```

[*Remove ads*](/account/join/)

## 概述和其他资源

在本教程中，您已经看到了如何安装 SpeechRecognition 包并使用它的`Recognizer`类来轻松识别来自文件(使用`record()`)和麦克风输入(使用`listen().`)的语音。您还看到了如何使用`record()`方法的`offset`和`duration`关键字参数来处理音频文件的片段。

您已经看到了噪声对转录准确性的影响，并且已经学习了如何使用`adjust_for_ambient_noise().`调整`Recognizer`实例对环境噪声的敏感度，还学习了`Recognizer`实例可能抛出哪些异常——对于糟糕的 API 请求使用`RequestError`,对于难以理解的语音使用`UnkownValueError`——以及如何使用`try...except`块处理这些异常。

语音识别是一个很深的课题，您在这里学到的知识只是皮毛。如果您有兴趣了解更多，这里有一些额外的资源。

**免费奖励:** [单击此处下载一个 Python 语音识别示例项目，该项目具有完整的源代码](#)，您可以将其用作自己的语音识别应用程序的基础。

有关演讲识别包的更多信息，请访问:

*   [图书馆参考](https://github.com/Uberi/speech_recognition/blob/master/reference/library-reference.rst)
*   [例题](https://github.com/Uberi/speech_recognition/tree/master/examples)
*   [故障排除页面](https://github.com/Uberi/speech_recognition#troubleshooting)

一些有趣的互联网资源:

*   麦克风背后:与计算机对话的科学。一部关于谷歌语音处理的短片。
*   语音识别的历史透视。美国计算机协会的通报(2014 年)。这篇文章对语音识别技术的发展进行了深入的学术探讨。
*   [语音识别技术的过去、现在和未来](https://medium.com/swlh/the-past-present-and-future-of-speech-recognition-technology-cf13c179aaf)Clark Boyd 在初创公司。这篇博客文章介绍了语音识别技术的概况，以及对未来的一些思考。

一些关于语音识别的好书:

*   [机器中的声音:构建理解语音的计算机](https://realpython.com/asins/0262533294/)，Pieraccini，麻省理工学院出版社(2012)。一本通俗易懂的大众读物，涵盖了语音处理的历史和现代进展。
*   [语音识别基础](https://realpython.com/asins/0130151572/)，Rabiner 和 Juang，Prentice Hall (1993)。贝尔实验室的研究人员 Rabiner 在设计第一批商业上可行的语音识别器方面发挥了重要作用。这本书已经有 20 多年的历史了，但是很多基本原理还是一样的。
*   [自动语音识别:一种深度学习的方法](https://realpython.com/asins/1447157788/)，于邓，施普林格(2014)。俞和邓是微软公司的研究人员，他们都在语音处理领域非常活跃。这本书涵盖了许多现代方法和前沿研究，但不适合数学胆小的人。

## 附录:识别非英语语言的语音

在本教程中，我们一直在识别英语语音，这是 SpeechRecognition 软件包中每个`recognize_*()`方法的默认语言。然而，识别其他语言的语音是完全可能的，而且很容易实现。

要识别不同语言的语音，请将`recognize_*()`方法的`language`关键字参数设置为对应于所需语言的字符串。大多数方法都接受 BCP-47 语言标签，比如`'en-US'`代表美式英语，或者`'fr-FR'`代表法语。例如，以下代码识别音频文件中的法语语音:

```py
import speech_recognition as sr

r = sr.Recognizer()

with sr.AudioFile('path/to/audiofile.wav') as source:
    audio = r.record(source)

r.recognize_google(audio, language='fr-FR')
```

只有以下方法接受`language`关键字参数:

*   `recognize_bing()`
*   `recognize_google()`
*   `recognize_google_cloud()`
*   `recognize_ibm()`
*   `recognize_sphinx()`

要找出你正在使用的 API 支持哪些语言标签，你必须查阅相应的[文档](https://github.com/Uberi/speech_recognition/blob/master/reference/library-reference.rst)。`recognize_google()`接受的标签列表可以在[这个栈溢出回答](https://stackoverflow.com/questions/14257598/what-are-language-codes-in-chromes-implementation-of-the-html5-speech-recogniti)中找到。

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和文字教程一起看，加深理解: [**用 Python 进行语音识别**](/courses/speech-recognition-python/)*********