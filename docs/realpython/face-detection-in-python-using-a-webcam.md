# 使用网络摄像头在 Python 中进行人脸检测

> 原文：<https://realpython.com/face-detection-in-python-using-a-webcam/>

*本教程是 Python* *中* *[人脸识别的后续，所以确保你已经过了第一篇帖子。](https://realpython.com/face-recognition-with-python/)*

正如在第一篇文章中提到的，从检测图像中的人脸到通过网络摄像头检测视频中的人脸非常容易——这正是我们将在这篇文章中详细介绍的。

> 在评论区提问之前:
> 
> 1.  不要跳过博客文章并试图运行代码。您必须理解代码的作用，不仅要正确运行它，还要对它进行故障排除。
> 2.  确保使用 OpenCV v2。
> 3.  您需要一个正常工作的网络摄像头，此脚本才能正常工作。
> 4.  查看其他评论/问题，因为您的问题可能已经得到解答。
> 
> 谢谢你。

**免费奖励:** ，向您展示真实世界 Python 计算机视觉技术的实用代码示例。

**注意:**也可以查看我们的[更新教程，使用 Python 进行人脸检测](https://realpython.com/traditional-face-detection-python/)。

## 先决条件

1.  OpenCV 已安装(详见之前的博文)
2.  正常工作的网络摄像头

[*Remove ads*](/account/join/)

## 代码

让我们直接进入从这个[库](https://github.com/shantnu/Webcam-Face-Detect)中取出的代码。

```py
import cv2
import sys

cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
```

现在我们来分解一下…

```py
import cv2
import sys

cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)
```

这个你应该很熟悉。我们正在创建一个面部层叠，就像我们在图像示例中所做的那样。

```py
video_capture = cv2.VideoCapture(0)
```

这一行将视频源设置为默认的网络摄像头，OpenCV 可以很容易地捕捉到。

> **注意**:你也可以在这里提供一个文件名，Python 会读入视频文件。然而，你需要安装 [ffmpeg](https://www.ffmpeg.org/) ，因为 OpenCV 本身不能解码压缩视频。Ffmpeg 充当 OpenCV 的前端，理想情况下，它应该直接编译到 OpenCV 中。这并不容易做到，尤其是在 Windows 上。

```py
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
```

在这里，我们捕捉视频。`read()`函数从视频源读取一帧，在本例中是网络摄像头。这将返回:

1.  实际视频帧读取(每个循环一帧)
2.  返回代码

返回代码告诉我们是否已经用完了帧，这在我们从文件中读取时会发生。当通过网络摄像头阅读时，这并不重要，因为我们可以永远记录，所以我们会忽略它。

```py
 # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)
```

同样，这段代码应该很熟悉。我们只是在捕捉的画面中寻找人脸。

```py
if cv2.waitKey(1) & 0xFF == ord('q'):
    break
```

我们等待按下“q”键。如果是，我们退出脚本。

```py
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
```

在这里，我们只是在清理。

## 测试！

[https://player.vimeo.com/video/100839478](https://player.vimeo.com/video/100839478)

所以，那是我手里拿着驾照。你可以看到这个算法同时跟踪真实的我和照片上的我。注意，我移动慢的时候，算法能跟上。但是，当我把我的手移到我的脸上时，它会混淆，把我的手腕误认为是一张脸。

就像我在上一篇文章中所说的，基于机器学习的算法很少是 100%准确的。我们还没有达到机械战警以每小时 100 英里的速度驾驶摩托车，用低质量的闭路电视摄像机追踪罪犯的阶段。

该代码逐帧搜索面部，因此需要相当大的处理能力。例如，在我用了五年的笔记本电脑上，它几乎占用了 90%的 CPU。

[*Remove ads*](/account/join/)

## 接下来的步骤

好吧，你知道如何识别人脸。但是，如果你想检测你自己的物体，比如你的汽车、电视或你最喜欢的玩具呢？

OpenCV 允许你创建自己的级联，但是这个过程并没有很好的记录。这里有一篇[博客](http://coding-robin.de/2013/07/22/train-your-own-opencv-haar-classifier.html)文章，向你展示如何训练你自己的级联来探测香蕉。

如果你想更进一步，识别每个人的脸——也许是在许多陌生人中发现和识别你的脸——这个任务是惊人的困难。这主要是由于涉及大量的图像预处理。但是，如果你愿意解决这个挑战，通过使用机器学习算法是可能的，如这里描述的。

**免费奖励:** ，向您展示真实世界 Python 计算机视觉技术的实用代码示例。

## 想了解更多？

在我接下来的课程中，将会更详细地介绍这一点，以及许多计算科学和机器学习的主题。该课程基于一个非常成功的 Kickstarter。

Kickstarter 已经结束，但你仍然可以在 [Python for Engineers](http://pythonforengineers.com/) 上订购课程。请访问了解更多信息。

此外，请在下面张贴您的视频链接，以获得我的直接反馈。有问题就评论。

哦——下次我们将讨论一些运动检测。敬请期待！**