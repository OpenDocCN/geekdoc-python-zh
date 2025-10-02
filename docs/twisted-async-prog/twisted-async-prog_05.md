# 由 Twisted 扶持的客户端

### 抽象地构建客户端

在第四部分中，我们构建了第一个使用 Twisted 的客户端。它确实能很好地工作，但仍有提高的空间。

首先是，这个客户端竟然有创建网络端口并接收端口处的数据这样枯燥的代码。Twisted 理应为我们实现这些例程性功能，省得我们每次写一个新的程序时都要自己去实现。这样做特别有用，可以将我们从异步 I/O 涉及的一些棘手的异常处理中解放出来(参看前面的[客户端](http://github.com/jdavisp3/twisted-intro/blob/master/twisted-client-1/get-poetry.py)) , 如果要跨平台就涉及到更多更加棘手的细节。如果你哪天下午有空，可以翻翻 Twisted 的 WIN32 实现源代码，看看里面有多少小针线是来处理跨平台的。

另一问题是与错误处理有关。当运行版本 1 的 Twisted 客户端从并没有提供服务的端口上下载诗歌时，它就会崩溃。当然我们是可以修正这个错误，但通过下面我们要介绍 Twisted 的 APIs 来处理这些类型的错误会更简单。

最后，那个客户端也不能复用。如果有另一个模块需要通过我们的客户端下载诗歌呢？人家怎么知道你的诗歌已经下载完毕？我们不能用一个方法简单地将一首诗下载完成后再传给人家，而在之前让人家处于等待状态。这确实是一个问题，但我们不准备在这个部分解决这个问题—在未来的部分中一定会解决这个问题。

我们将会使用一些高层次的 APIs 和接口来解决第一、二个问题。Twisted 框架是由众多抽象层松散地组合起来的。因此，学习 Twisted 也就意味着需要学习这些层都提供什么功能，例如每层都有哪些 APIs，接口和实例可供使用。接下来我们会通过剖析 Twisted 最最重要的部分来更好地感受一下 Twisted 都是怎么组织的。一旦你对 Twisted 的整个结构熟悉了，学习新的部分会简单多了。

一般来说，每个 Twisted 的抽象都只与一个特定的概念相关。例如，第四部分中的客户端使用的 IReadDescriptor，它就是"一个可以读取字节的文件描述符"的抽象。一个抽象往往会通过定义接口来指定那些想实现这个抽象（也就是实现这个接口）的对象的形为。在学习新的 Twisted 抽象概念时，最需要谨记的就是：

> **多数高层次抽象都是在低层次抽象的基础上建立的，很少有另立门户的。**

因此，你在学习新的 Twisted 抽象概念时，始终要记住它做什么和不做什么。特别是，如果一个早期的抽象 A 实现了 F 特性，那么 F 特性不太可能再由其它任何抽象来实现。另外，如果另外一个抽象需要 F 特性，那么它会使用 A 而不是自己再去实现 F。（通常的做法，B 可能会通过继承 A 或获得一个指向 A 实例的引用）

网络非常的复杂，因此 Twisted 包含很多抽象的概念。通过从低层的抽象讲起，我们希望能更清楚起看到在一个 Twisted 程序中各个部分是怎么组织起来的。

### 核心的循环体

第一个我们要学习的抽象，也是 Twisted 中最重要的，就是 reactor。在每个通过 Twisted 搭建起来的程序中心处，不管你这个程序有多少层，总会有一个 reactor 循环在不停止地驱动程序的运行。再也没有比 reactor 提供更加基础的支持了。实际上，Twisted 的其它部分（即除了 reactor 循环体）可以这样理解：它们都是来辅助 X 来更好地使用 reactor，这里的 X 可以是提供 Web 网页、处理一个数据库查询请求或其它更加具体的内容。尽管坚持像上一个客户端一样使用低层 APIs 是可能的，但如果我们执意那样做，那么我们必需自己来实现非常多的内容。而在更高的层次上，意味着我们可以少写很多代码。

但是当在外层思考与处理问题时, 很容易就忘记了 reactor 的存在了。在任何一个常见大小的 Twisted 程序中 ，确实很少会有直接与 reactor 的 APIs 交互。低层的抽象也是一样（即我们很少会直接与其交互）。我们在上一个客户端中用到的文件描述符抽象，就被更高层的抽象更好的归纳以至于我们很少会在真正的 Twisted 程序中遇到。（他们在内部依然在被使用，只是我们看不到而已）

至于文件描述符抽象的消息，这并不是一个问题。让 Twisted 掌舵异步 I/O 处理，这样我们就可以更加关注我们实际要解决的问题。但对于 reactor 不一样，它永远都不会消失。当你选择使用 Twisted，也就意味着你选择使用 Reactor 模式，并且意味着你需要使用回调与多任务合作的"交互式"编程方式。如果你想正确地使用 Twisted，你必须牢记 reactor 的存在。我们将在第六部分更加详细的讲解部分内容。但是现在要强调的是：

> **图 5 与图 6 是这个系列中最最重要的图**

我们还将用图来描述新的概念，但这两个图是需要你牢记在脑海中的。可以这样说，我在写 Twisted 程序时一直想着这两张图。

在我们付诸于代码前，有三个新的概念需要阐述清楚：Transports, Protocols, Protocol Factories

### Transports

Transports 抽象是通过 Twisted 中 interfaces 模块中 ITransport 接口定义的。一个 Twisted 的 Transport 代表一个可以收发字节的单条连接。对于我们的诗歌下载客户端而言，就是对一条 TCP 连接的抽象。但是 Twisted 也支持诸如 Unix 中管道和 UDP。Transport 抽象可以代表任何这样的连接并为其代表的连接处理具体的异步 I/O 操作细节。

如果你浏览一下 ITransport 中的方法，可能找不到任何接收数据的方法。这是因为 Transports 总是在低层完成从连接中异步读取数据的许多细节工作，然后通过回调将数据发给我们。相似的原理，Transport 对象的写相关的方法为避免阻塞也不会选择立即写我们要发送的数据。告诉一个 Transport 要发送数据，只是意味着：尽快将这些数据发送出去，别产生阻塞就行。当然，数据会按照我们提交的顺序发送。

通常我们不会自己实现一个 Transport。我们会去使用 Twisted 提供的实现类，即在传递给 reactor 时会为我们创建一个对象实例。

### Protocols

Twisted 的 Protocols 抽象由 interfaces 模块中的 IProtocol 定义。也许你已经想到，Protocol 对象实现协议内容。也就是说，一个具体的 Twisted 的 Protocol 的实现应该对应一个具体网络协议的实现，像 FTP、IMAP 或其它我们自己制定的协议。我们的诗歌下载协议，正如它表现的那样，就是在连接建立后将所有的诗歌内容全部发送出去并且在发送完毕后关闭连接。

严格意义上讲，每一个 Twisted 的 Protocols 类实例都为一个具体的连接提供协议解析。因此我们的程序每建立一条连接（对于服务方就是每接受一条连接），都需要一个协议实例。这就意味着，Protocol 实例是存储协议状态与间断性（由于我们是通过异步 I/O 方式以任意大小来接收数据的）接收并累积数据的地方。

因此，Protocol 实例如何得知它为哪条连接服务呢？如果你阅读 IProtocol 定义会发现一个 makeConnection 函数。这是一个回调函数，Twisted 会在调用它时传递给其一个也是仅有的一个参数，即 Transport 实例。这个 Transport 实例就代表 Protocol 将要使用的连接。

Twisted 内置了很多实现了通用协议的 Protocol。你可以在[twisted.protocols.basic](http://twistedmatrix.com/trac/browser/trunk/twisted/protocols/basic.py)中找到一些稍微简单点的。在你尝试写新 Protocol 时，最好是看看 Twisted 源码是不是已经有现成的存在。如果没有，那实现一个自己的协议是非常好的，正如我们为诗歌下载客户端做的那样。

### Protocol Factories

因此每个连接需要一个自己的 Protocol，而且这个 Protocol 是我们自己定义的类的实例。由于我们会将创建连接的工作交给 Twisted 来完成，Twisted 需要一种方式来为一个新的连接创建一个合适的协议。创建协议就是 Protocol Factories 的工作了。

也许你已经猜到了，Protocol Factory 的 API 由[IProtocolFactory](http://twistedmatrix.com/trac/browser/trunk/twisted/internet/interfaces.py)来定义，同样在[interfaces](http://twistedmatrix.com/trac/browser/trunk/twisted/internet/interfaces.py)模块中。Protocol Factory 就是 Factory 模式的一个具体实现。buildProtocol 方法在每次被调用时返回一个新 Protocol 实例，它就是 Twisted 用来为新连接创建新 Protocol 实例的方法。

### 诗歌下载客户端 2.0：第一滴心血

好吧，让我们来看看由 Twisted 支持的诗歌下载客户端 2.0。源码可以在这里[twisted-client-2/get-poetry.py](http://github.com/jdavisp3/twisted-intro/blob/master/twisted-client-2/get-poetry.py)。你可以像前面一样运行它，并得到相同的输出。这也是最后一个在接收到数据时打印其任务的客户端版本了。到现在为止，对于所有 Twisted 程序都是交替执行任务并处理相对较少数量数据的，应该很清晰了。我们依然通过 print 函数来展示在关键时刻在进行什么内容，但将来客户端不会在这样繁锁。

在第二个版本中，sockets 不会再出现了。我们甚至不需要引入 socket 模块也不用引用 socket 对象和文件描述符。取而代之的是，我们告诉 reactor 来创建到诗歌服务器的连接，代码如下面所示：

```py
factory = PoetryClientFactory(len(addresses))

from twisted.internet import reactor

for address in addresses:
    host, port = address
    reactor.connectTCP(host, port, factory) 
```

我们需要关注的是 connectTCP 这个函数。前两个参数的含义很明显，不解释了。第三个参数是我们自定义的 PoetryClientFactory 类的实例对象。这是一个专门针对诗歌下载客户端的 Protocol Factory，将它传递给 reactor 可以让 Twisted 为我们创建一个 PoetryProtocol 实例。

值得注意的是，从一开始我们既没有实现 Factory 也没有去实现 Protocol，不像在前面那个客户端中我们去实例化我们 PoetrySocket 类。我们只是继承了 Twisted 在 twisted.internet.protocol 中提供的基类。Factory 的基类是 twisted.internet.protocol.Factory，但我们使用客户端专用（即不像服务器端那样监听一个连接，而是主动创建一个连接）的 ClientFactory 子类来继承。

我们同样利用了 Twisted 的 Factory 已经实现了 buildProtocol 方法这一优势来为我们所用。我们要在子类中调用基类中的实现：

```py
def buildProtocol(self, address):
    proto = ClientFactory.buildProtocol(self, address)
    proto.task_num = self.task_num
    self.task_num += 1
    return proto 
```

基类怎么会知道我们要创建什么样的 Protocol 呢？注意，我们的 PoetryClientFactory 中有一个 protocol 类变量：

```py
class PoetryClientFactory(ClientFactory):

    task_num = 1

    protocol = PoetryProtocol # tell base class what proto to build 
```

基类 Factory 实现 buildProtocol 的过程是：安装（创建一个实例）我们设置在 protocol 变量上的 Protocol 类与在这个实例（此处即 PoetryProtocol 的实例）的 factory 属性上设置一个产生它的 Factory 的引用（此处即实例化 PoetryProtocol 的 PoetryClientFactory）。这个过程如图 8 所示：

![Protocol 的生成过程](img/p05_protocols-1.png "Protocol 的生成过程")图 8 Protocol 的生成过程

正如我们提到的那样，位于 Protocol 对象内的 factory 属性字段允许在都由同一个 factory 产生的 Protocol 之间共享数据。由于 Factories 都是由用户代码来创建的（即在用户的控制中），因此这个属性也可以实现 Protocol 对象将数据传递回一开始初始化请求的代码中来，这将在第六部分看到。

值得注意的是，虽然在 Protocol 中有一个属性指向生成其的 Protocol Factory，在 Factory 中也有一个变量指向一个 Protocol 类，但通常来说，一个 Factory 可以生成多个 Protocol。

在 Protocol 创立的第二步便是通过 makeConnection 与一个 Transport 联系起来。我们无需自己来实现这个函数而使用 Twisted 提供的默认实现。默认情况是，makeConnection 将 Transport 的一个引用赋给（Protocol 的）transport 属性，同时置（同样是 Protocol 的）connected 属性为 True，正如图 9 描述的一样：

![Protocol 遇到其 Transport](img/p05_protocols-2.png "Protocol 遇到其 Transport")图 9 Protocol 遇到其 Transport

一旦初始化到这一步后，Protocol 开始其真正的工作—将低层的数据流翻译成高层的协议规定格式的消息。处理接收到数据的主要方法是 dataReceived，我们的客户端是这样实现的：

```py
def dataReceived(self, data):
    self.poem += data
    msg = 'Task %d: got %d bytes of poetry from %s'
    print  msg % (self.task_num, len(data), self.transport.getHost()) 
```

每次 dateReceved 被调用就意味着我们得到一个新字符串。由于与异步 I/O 交互，我们不知道能接收到多少数据，因此将接收到的数据缓存下来直到完成一个完整的协议规定格式的消息。在我们的例子中，诗歌只有在连接关闭时才下载完毕，因此我们只是不断地将接收到的数据添加到我们的.poem 属性字段中。

注意我们使用了 Transport 的 getHost 方法来取得数据来自的服务器信息。我们这样做只是与前面的客户端保持一致。相反，我们的代码没有必要这样做，因为我们没有向服务器发送任何消息，也就没有必要知道服务器的信息了。

我们来看一下 dataReceved 运行时的快照。在 2.0 版本相同的目录下有一个 twisted-client-2/get-poetry-stack.py。它与 2.0 版本的不同之处只在于：

```py
def dataReceived(self, data):
    traceback.print_stack()
    os._exit(0) 
```

这样一改，我们就能打印出跟踪堆栈的信息，然后离开程序，可以用下面的命令来运行它：

```py
python twisted-client-2/get-poetry-stack.py 10000 
```

你会得到内容如下的跟踪堆栈：

```py
File "twisted-client-2/get-poetry-stack.py", line 125, in
    poetry_main()

... # I removed a bunch of lines here

File ".../twisted/internet/tcp.py", line 463, in doRead  # Note the doRead callback
    return self.protocol.dataReceived(data)
File "twisted-client-2/get-poetry-stack.py", line 58, in dataReceived
    traceback.print_stack() 
```

看见没，有我们在 1.0 版本客户端的 doRead 回调函数。我们前面也提到过，Twisted 在建立新抽象层会使用已有的实现而不是另起炉灶。因此必然会有一个 IReadDescriptor 的实例在辛苦的工作，它是由 Twisted 代码而非我们自己的代码来实现。如果你表示怀疑，那么就看看 twisted.internet.tcp 中的实现吧。如果你浏览代码会发现，由同一个类实现了 IWriteDescriptor 与 ITransport。因此 **IReadDescriptor 实际上就是变相的 Transport 类**。可以用图 10 来形象地说明 dateReceived 的回调过程：

![dateReceived 回调过程 ](img/p05_reactor-data-received.png "dateReceived 回调过程 ")图 10 dateReceived 回调过程

一旦诗歌下载完成，PoetryProtocol 就会通知它的 PooetryClientFactory：

```py
def connectionLost(self, reason):     
  self.poemReceived(self.poem) 
def poemReceived(self, poem):    
  self.factory.poem_finished(self.task_num, poem) 
```

当 transport 的连接关闭时，conncetionLost 回调会被激活。reason 参数是一个[twisted.python.failure.Failure](http://twistedmatrix.com/trac/browser/trunk/twisted/python/failure.py)的实例对象，其携带的信息能够说明连接是被安全的关闭还是由于出错被关闭的。我们的客户端因认为总是能完整地下载完诗歌而忽略了这一参数。

工厂会在所有的诗歌都下载完毕后关闭 reactor。再次重申：我们代码的工作就是用来下载诗歌-这意味我们的 PoetryClientFactory 缺少复用性。我们将在下一部分修正这一缺陷。值得注意的是，poem_finish 回调函数是如何通过跟踪剩余诗歌数的：

```py
 ...
    self.poetry_count -= 1

    if self.poetry_count == 0:
    ... 
```

如果我们采用多线程以让每个线程分别下载诗歌，这样我们就必须使用一把锁来管理这段代码以免多个线程在同一时间调用 poem_finish。但是在交互式体系下就不必担心了。由于 reactor 只能一次启用一个回调。

新的客户端实现在处理错误上也比先前的优雅的多，下面是 PoetryClientFactory 处理错误连接的回调实现代码：

```py
def clientConnectionFailed(self, connector, reason):
    print 'Failed to connect to:', connector.getDestination()
    self.poem_finished() 
```

注意，回调是在工厂内部而不是协议内部实现。由于协议是在连接建立后才创建的，而工厂能够在连接未能成功建立时捕获消息。

### 结束语

版本 2 的客户端使用的抽象对于那些 Twisted 高手应该非常熟悉。如果仅仅是为在命令行上打印出下载的诗歌这个功能，那么我们已经完成了。但如果想使我们的代码能够复用，能够被内嵌在一些包含诗歌下载功能并可以做其它事情的大软件中，我们还有许多工作要做，我们将在第六部分讲解相关内容。

### 参考

本部分原作参见: dave @ [`krondo.com/?p=1522`](http://krondo.com/?p=1522)

本部分翻译内容参见杨晓伟的博客 [`blog.sina.com.cn/s/blog_704b6af70100q2ac.html`](http://blog.sina.com.cn/s/blog_704b6af70100q2ac.html)