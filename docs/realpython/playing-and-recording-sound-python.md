# 用 Python 播放和录制声音

> 原文：<https://realpython.com/playing-and-recording-sound-python/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。配合文字教程一起看，加深理解: [**用 Python**](/courses/playing-and-recording-sound-python/) 演奏录音

如果您想使用 Python 来播放或录制声音，那么您来对地方了！在本教程中，您将学习如何使用一些最流行的音频库在 Python 中播放和录制声音。您将首先了解播放和录制声音的最直接的方法，然后了解一些提供更多功能的库，以换取几行额外的代码。

**本教程结束时，你将知道如何:**

*   播放 MP3 和 WAV 文件，以及一系列其他音频格式
*   播放包含声音的 NumPy 和 Python 数组
*   使用 Python 录制声音
*   以一系列不同的文件格式保存您的录音或音频文件

要获得与音频相关的 Python 库的完整列表，请查看 Python 中关于音频的[维基页面。](https://wiki.python.org/moin/Audio/)

**免费下载:** [从 Python 技巧中获取一个示例章节:这本书](https://realpython.com/bonus/python-tricks-sample-pdf/)用简单的例子向您展示了 Python 的最佳实践，您可以立即应用它来编写更漂亮的+Python 代码。

## 播放音频文件

下面，您将看到如何使用一系列 Python 库来播放音频文件。其中一些库允许您播放一系列音频格式，包括 MP3 和 NumPy 数组。以下所有库都允许您播放 WAV 文件，有些库比其他库多几行代码:

*   如果你只是想播放一个 WAV 或 MP3 文件，那么 **`playsound`** 是最简单的软件包。除了简单的回放，它不提供任何功能。

*   **`simpleaudio`** 让你播放 WAV 文件和 NumPy 数组，并给你检查文件是否还在播放的选项。

*   **`winsound`** 允许你播放 WAV 文件或者让你的扬声器发出嘟嘟声，但是它只在 Windows 上有效。

*   **`python-sounddevice`****`pyaudio`**为 PortAudio 库提供绑定，用于跨平台播放 WAV 文件。

*   **`pydub`** 需要 **`pyaudio`** 进行音频播放，但安装了`ffmpeg`后，只需几行代码就能让你播放大范围的音频格式。

让我们一个一个的来看看这些用于音频播放的库。

[*Remove ads*](/account/join/)

### `playsound`

[`playsound`](https://pypi.org/project/playsound/) 是一个[“纯 Python，跨平台，单一功能模块，对播放声音没有依赖性。”](https://pypi.org/project/playsound/)使用这个模块，你可以用一行代码播放一个声音文件:

```py
from playsound import playsound

playsound('myfile.wav')
```

`playsound`的[文档](https://pypi.org/project/playsound/)声明它已经在 WAV 和 MP3 文件上测试过，但它也可能适用于其他文件格式。

这个库最后一次更新是在 2017 年 6 月。在撰写本文时，它似乎工作得很好，但不清楚它是否仍然支持较新的 Python 版本。

### `simpleaudio`

[`simpleaudio`](https://simpleaudio.readthedocs.io/en/latest/) 是一个跨平台的库，用于播放(单声道和立体声)WAV 文件，不依赖。以下代码可用于播放 WAV 文件，并在终止脚本之前等待文件播放完毕:

```py
import simpleaudio as sa

filename = 'myfile.wav'
wave_obj = sa.WaveObject.from_wave_file(filename)
play_obj = wave_obj.play()
play_obj.wait_done()  # Wait until sound has finished playing
```

WAV 文件包含一个代表原始音频数据的位序列，以及 RIFF(资源交换文件格式)格式的带有元数据的**头。**

对于 CD 录音，行业标准是将每个音频样本(与气压相关的单个音频数据点)存储为一个 **16 位**值，每秒 **44100** 个样本。

为了减小文件大小，以较低的采样率(例如每秒 8000 个样本)存储一些记录(例如人类语音)可能就足够了，尽管这确实意味着较高的声音频率可能不会被准确地表示。

本教程中讨论的一些库播放和录制 **`bytes`** 对象，而其他的使用 **NumPy 数组**来存储原始音频数据。

两者都对应于一系列数据点，为了播放声音，可以以指定的采样速率回放这些数据点。对于一个 **`bytes`** 对象，每个样本存储为一组两个 8 位值，而在 NumPy 数组中，每个元素可以包含一个对应于单个样本的 16 位值。

这两种数据类型的一个重要区别是， **`bytes`对象是不可变的**，而 **NumPy 数组是可变的**，这使得后者更适合生成声音和更复杂的信号处理。关于如何使用 NumPy 的更多信息，请看我们的 [NumPy 教程](https://realpython.com/tutorials/numpy/)。

`simpleaudio`允许你使用`simpleaudio.play_buffer()`来玩 NumPy 和 Python 数组以及`bytes`对象。确保您已经安装了 NumPy，以便下面的示例以及`simpleaudio`能够工作。(安装了`pip`之后，你可以通过从你的控制台运行`pip install numpy`来完成这个任务。)

关于如何使用`pip`安装包的更多信息，请看一下 [Pipenv:新 Python 打包工具](https://realpython.com/pipenv-guide/)指南。

在下面，您将看到如何生成一个对应于 440 赫兹音调的 NumPy 数组，并使用`simpleaudio.play_buffer()`进行回放:

```py
import numpy as np
import simpleaudio as sa

frequency = 440  # Our played note will be 440 Hz
fs = 44100  # 44100 samples per second
seconds = 3  # Note duration of 3 seconds

# Generate array with seconds*sample_rate steps, ranging between 0 and seconds
t = np.linspace(0, seconds, seconds * fs, False)

# Generate a 440 Hz sine wave
note = np.sin(frequency * t * 2 * np.pi)

# Ensure that highest value is in 16-bit range
audio = note * (2**15 - 1) / np.max(np.abs(note))
# Convert to 16-bit data
audio = audio.astype(np.int16)

# Start playback
play_obj = sa.play_buffer(audio, 1, 2, fs)

# Wait for playback to finish before exiting
play_obj.wait_done()
```

接下来，让我们看看如何使用`winsound`在 Windows 机器上播放 WAV 文件。

### `winsound`

如果你使用 Windows，你可以使用内置的 [`winsound`](https://docs.python.org/3.6/library/winsound.html) 模块来访问其基本的声音播放机制。播放 WAV 文件只需几行代码:

```py
import winsound

filename = 'myfile.wav'
winsound.PlaySound(filename, winsound.SND_FILENAME)
```

`winsound`不支持播放除 WAV 文件以外的任何文件。它确实允许你使用`winsound.Beep(frequency, duration)`让你的扬声器发出嘟嘟声。例如，您可以使用以下代码发出 1000 赫兹的提示音 100 毫秒:

```py
import winsound

winsound.Beep(1000, 100)  # Beep at 1000 Hz for 100 ms
```

接下来，您将学习如何使用`python-sounddevice`模块进行跨平台音频播放。

[*Remove ads*](/account/join/)

### `python-sounddevice`

正如其文档中所述， [`python-sounddevice`](https://python-sounddevice.readthedocs.io/en/latest/) “为 PortAudio 库提供绑定和一些方便的函数来播放和录制包含音频信号的 NumPy 数组”。为了播放 WAV 文件，需要安装`numpy`和`soundfile`，以 NumPy 数组的形式打开 WAV 文件。

安装了`python-sounddevice`、`numpy`和`soundfile`之后，您现在可以将 WAV 文件作为 NumPy 数组读取并回放:

```py
import sounddevice as sd
import soundfile as sf

filename = 'myfile.wav'
# Extract data and sampling rate from file
data, fs = sf.read(filename, dtype='float32')  
sd.play(data, fs)
status = sd.wait()  # Wait until file is done playing
```

包含`sf.read()`的行提取原始音频数据，以及存储在其 RIFF 头中的文件的采样率，`sounddevice.wait()`确保脚本仅在声音结束播放后终止。

接下来，我们将学习如何使用`pydub`来播放声音。安装了正确的依赖项后，它允许你播放各种各样的音频文件，并且比`python-soundevice`提供了更多的音频处理选项。

### `pydub`

虽然 [`pydub`](https://www.github.com/jiaaro/pydub) 可以打开保存 WAV 文件，没有任何依赖关系，但是需要安装音频播放包才能播放音频。强烈推荐`simpleaudio`，但是`pyaudio`、`ffplay`、`avplay`也是备选方案。

下面的代码可以用来播放带有`pydub`的 WAV 文件:

```py
from pydub import AudioSegment
from pydub.playback import play

sound = AudioSegment.from_wav('myfile.wav')
play(sound)
```

为了播放其他音频类型，如 MP3 文件，应安装 [`ffmpeg`](https://www.ffmpeg.org/download.html) 或 [`libav`](https://libav.org/download/) 。查看`pydub`的[文档](https://github.com/jiaaro/pydub#getting-ffmpeg-set-up)获取说明。作为文档中描述的步骤的替代， [`ffmpeg-python`](https://github.com/kkroening/ffmpeg-python) 为`ffmpeg`提供绑定，可以使用 pip 进行安装:

```py
$ pip install ffmpeg-python
```

安装了`ffmpeg`之后，回放一个 MP3 文件只需要对我们之前的代码做一点小小的改动:

```py
from pydub import AudioSegment
from pydub.playback import play

sound = AudioSegment.from_mp3('myfile.mp3')
play(sound)
```

使用`AudioSegment.from_file(filename, filetype)`结构，你可以播放任何类型的`ffmpeg`支持的[音频文件。例如，您可以使用以下内容播放 WMA 文件:](http://www.ffmpeg.org/general.html#File-Formats)

```py
sound = AudioSegment.from_file('myfile.wma', 'wma')
```

除了回放声音文件，`pydub`还允许您以不同的文件格式保存音频(稍后将详细介绍)，分割音频，计算音频文件的长度，淡入或淡出，以及应用交叉淡入淡出。

`AudioSegment.reverse()`创建倒放的音频片段的副本，文档描述为[“对平克·弗洛伊德、鬼混和一些音频处理算法有用。”](https://github.com/jiaaro/pydub/blob/master/API.markdown)

### `pyaudio`

[`pyaudio`](https://people.csail.mit.edu/hubert/pyaudio/) 为跨平台音频 I/O 库 PortAudio 提供绑定。这意味着您可以使用`pyaudio`在各种平台上播放和录制音频，包括 Windows、Linux 和 Mac。使用`pyaudio`，通过写入`.Stream`来播放音频:

```py
import pyaudio
import wave

filename = 'myfile.wav'

# Set chunk size of 1024 samples per data frame
chunk = 1024  

# Open the sound file 
wf = wave.open(filename, 'rb')

# Create an interface to PortAudio
p = pyaudio.PyAudio()

# Open a .Stream object to write the WAV file to
# 'output = True' indicates that the sound will be played rather than recorded
stream = p.open(format = p.get_format_from_width(wf.getsampwidth()),
                channels = wf.getnchannels(),
                rate = wf.getframerate(),
                output = True)

# Read data in chunks
data = wf.readframes(chunk)

# Play the sound by writing the audio data to the stream
while data != '':
    stream.write(data)
    data = wf.readframes(chunk)

# Close and terminate the stream
stream.close()
p.terminate()
```

您可能已经注意到，用`pyaudio`播放声音比用您之前看到的库播放声音要复杂一些。这意味着，如果您只想在 Python 应用程序中播放声音效果，它可能不是您的首选。

然而，因为`pyaudio`给了你更多的底层控制，所以可以获取和设置你的输入和输出设备的参数，并检查你的 CPU 负载和输入或输出延迟。

它还允许您在回调模式下播放和录制音频，在回调模式下，当需要新数据进行回放或可用于录制时，会调用指定的回调函数。如果您的音频需求不仅仅是简单的回放，这些选项使`pyaudio`成为一个合适的库。

既然您已经看到了如何使用许多不同的库来播放音频，那么是时候看看如何使用 Python 来自己录制音频了。

[*Remove ads*](/account/join/)

## 录制音频

[`python-sounddevice`](https://python-sounddevice.readthedocs.io/en/latest/) 和 [`pyaudio`](https://people.csail.mit.edu/hubert/pyaudio/) 库提供了用 Python 录制音频的方法。`python-sounddevice`记录到 NumPy 数组，`pyaudio`记录到`bytes`对象。这两个都可以分别使用`scipy`和`wave`库存储为 WAV 文件。

### `python-sounddevice`

[`python-sounddevice`](https://python-sounddevice.readthedocs.io/en/latest/) 允许你从你的麦克风记录音频，并将其存储为一个 NumPy 数组。这是一种方便的声音处理数据类型，可以使用 [`scipy.io.wavfile`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.write.html) 模块将其转换为 WAV 格式进行存储。确保为以下示例(`pip install scipy`)安装`scipy`模块。这将自动安装 NumPy 作为其依赖项之一:

```py
import sounddevice as sd
from scipy.io.wavfile import write

fs = 44100  # Sample rate
seconds = 3  # Duration of recording

myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
sd.wait()  # Wait until recording is finished
write('output.wav', fs, myrecording)  # Save as WAV file
```

### `pyaudio`

[在本文前面的](#pyaudio)中，你通过阅读一首 [`pyaudio.Stream()`](https://people.csail.mit.edu/hubert/pyaudio/docs/#class-stream) 学会了如何弹奏声音。可以通过写入此流来录制音频:

```py
import pyaudio
import wave

chunk = 1024  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 2
fs = 44100  # Record at 44100 samples per second
seconds = 3
filename = "output.wav"

p = pyaudio.PyAudio()  # Create an interface to PortAudio

print('Recording')

stream = p.open(format=sample_format,
                channels=channels,
                rate=fs,
                frames_per_buffer=chunk,
                input=True)

frames = []  # Initialize array to store frames

# Store data in chunks for 3 seconds
for i in range(0, int(fs / chunk * seconds)):
    data = stream.read(chunk)
    frames.append(data)

# Stop and close the stream 
stream.stop_stream()
stream.close()
# Terminate the PortAudio interface
p.terminate()

print('Finished recording')

# Save the recorded data as a WAV file
wf = wave.open(filename, 'wb')
wf.setnchannels(channels)
wf.setsampwidth(p.get_sample_size(sample_format))
wf.setframerate(fs)
wf.writeframes(b''.join(frames))
wf.close()
```

现在您已经了解了如何使用`python-sounddevice`和`pyaudio`录制音频，您将学习如何将您的录音(或任何其他音频文件)转换为一系列不同的音频格式。

## 保存和转换音频

您[在前面的](#python-sounddevice)中看到，您可以使用`scipy.io.wavfile`模块将 NumPy 数组存储为 WAV 文件。 [`wavio`模块](https://pypi.org/project/wavio/)同样可以让你在 WAV 文件和 NumPy 数组之间转换。如果你想以不同的文件格式存储你的音频， [`pydub`](https://github.com/jiaaro/pydub) 和 [`soundfile`](https://pypi.org/project/SoundFile/) 就派上了用场，因为它们允许你读写一系列流行的文件格式(如 MP3、FLAC、WMA 和 FLV)。

### `wavio`

这个模块依赖于`numpy`并允许你读取 WAV 文件作为 NumPy 数组，并保存 NumPy 数组为 WAV 文件。

要将 NumPy 数组保存为 WAV 文件，可以使用`wavio.write()`:

```py
import wavio

wavio.write("myfile.wav", my_np_array, fs, sampwidth=2)
```

在这个例子中，`my_np_array`是包含音频的 NumPy 数组，`fs`是录音的采样率(通常为 44100 或 44800 Hz)，`sampwidth`是音频的采样宽度(每个样本的字节数，通常为 1 或 2 字节)。

### `soundfile`

[`soundfile`](https://pypi.org/project/SoundFile/) 库可以读写`libsndfile` 支持的所有[文件格式。虽然它不能播放音频，但它允许你在 FLAC、AIFF 和一些不太常见的音频格式之间来回转换。要将 WAV 文件转换为 FLAC，可以使用以下代码:](http://www.mega-nerd.com/libsndfile/#Features)

```py
import soundfile as sf

# Extract audio data and sampling rate from file 
data, fs = sf.read('myfile.wav') 
# Save as FLAC file at correct sampling rate
sf.write('myfile.flac', data, fs)
```

类似的代码也可以在`libsndfile`支持的其他文件格式之间转换。

[*Remove ads*](/account/join/)

### `pydub`

[`pydub`](https://github.com/jiaaro/pydub) 可以让你将音频保存为`ffmpeg`支持的[格式，几乎囊括了你日常生活中可能遇到的所有音频类型。例如，您可以使用以下代码将 WAV 文件转换为 MP3:](https://www.ffmpeg.org/general.html#File-Formats)

```py
from pydub import AudioSegment
sound = AudioSegment.from_wav('myfile.wav')

sound.export('myfile.mp3', format='mp3')
```

使用`AudioSegment.from_file()`是一种更通用的加载音频文件的方式。例如，如果您想将文件从 MP3 转换回 WAV，您可以执行以下操作:

```py
from pydub import AudioSegment
sound = AudioSegment.from_file('myfile.mp3', format='mp3')

sound.export('myfile.wav', format='wav')
```

这段代码应该适用于`ffmpeg`支持的任何音频文件格式[。](https://www.ffmpeg.org/general.html#File-Formats)

## 音频库对比

下表比较了本教程中讨论的库的功能:

| 图书馆 | 平台 | 重放 | 记录 | 皈依者 | 属国 |
| --- | --- | --- | --- | --- | --- |
| [T2`playsound`](https://pypi.org/project/playsound/) | 跨平台 | WAV，MP3 | - | - | 没有人 |
| [T2`simpleaudio`](https://simpleaudio.readthedocs.io/en/latest/) | 跨平台 | WAV，数组，`bytes` | - | - | 没有人 |
| [T2`winsound`](https://docs.python.org/3.6/library/winsound.html) | Windows 操作系统 | 声音资源文件 | - | - | 没有人 |
| [T2`sounddevice`](https://python-sounddevice.readthedocs.io/en/latest/) | 跨平台 | 数字阵列 | 数字阵列 | - | `numpy`，`soundfile` |
| [T2`pydub`]([pydub](https://github.com/jiaaro/pydub)) | 跨平台 | [`ffmpeg`](http://www.ffmpeg.org/general.html#File-Formats)支持的任何类型 | - | [`ffmpeg`](http://www.ffmpeg.org/general.html#File-Formats)支持的任何类型 | `simpleaudio`，`ffmpeg` |
| [T2`pyaudio`](https://people.csail.mit.edu/hubert/pyaudio/) | 跨平台 | `bytes` | `bytes` | - | `wave` |
| [T2`wavio`](https://pypi.org/project/wavio/) | 跨平台 | - | - | WAV，数字阵列 | `numpy`，`wave` |
| [T2`soundfile`](https://pypi.org/project/SoundFile/) | 跨平台 | - | - | [`libsndfile`](http://www.mega-nerd.com/libsndfile/#Features)支持的任何类型 | `CFFI`、`libsndfile`、`numpy` |

本教程中包含的库是根据它们的易用性和受欢迎程度而选择的。要获得更全面的 Python 音频库列表，请查看关于 Python 音频的 wiki 页面。

## 结论:在 Python 中播放和录制声音

在本教程中，您学习了如何使用一些最流行的音频库在 Python 中播放和录制音频。您还了解了如何以各种不同的格式保存音频。

您现在能够:

*   **播放**多种音频格式，包括 WAV、MP3 和 NumPy 阵列
*   **将麦克风中的**音频录制到 NumPy 或 Python 数组中
*   **储存**您录制的各种不同格式的音频，包括 WAV 和 MP3
*   将您的声音文件转换成一系列不同的音频格式

现在，您已经有了所需的信息，可以帮助您决定使用哪些库来开始在 Python 中处理音频。向前迈进，开发一些令人敬畏的音频应用程序！

*立即观看**本教程有真实 Python 团队创建的相关视频课程。配合文字教程一起看，加深理解: [**用 Python**](/courses/playing-and-recording-sound-python/) 演奏录音******