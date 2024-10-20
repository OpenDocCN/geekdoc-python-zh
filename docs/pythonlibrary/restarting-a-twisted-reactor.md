# 重启扭曲的反应堆

> 原文：<https://www.blog.pythonlibrary.org/2016/09/14/restarting-a-twisted-reactor/>

几周前我开始使用 twisted。对于不知道的人来说， [twisted](http://twistedmatrix.com/trac/) 是“用 Python 写的事件驱动联网引擎”。如果您以前从未做过异步编程，那么学习曲线会非常陡峭。在我工作的项目中，我遇到了一种情况，我认为我需要重新启动扭曲的反应堆。根据我在网上找到的一切，重启反应堆是不支持的。但是我很固执，所以我试着找到一种方法。

### 重启扭曲的反应堆

让我们从创建一个非常标准的 twisted 服务器开始。我们将子类化 **LineReceiver** ，它基本上是一个接受整行文本的 TCP 服务器，尽管它也可以处理原始数据。让我们看一下代码:

```py

from twisted.internet import reactor
from twisted.protocols.basic import LineReceiver
from twisted.internet.protocol import Factory

PORT = 9000

class LineServer(LineReceiver):

    def connectionMade(self):
        """
        Overridden event handler that is called when a connection to the 
        server was made
        """
        print "server received a connection!"

    def connectionLost(self, reason):
        """
        Overridden event handler that is called when the connection 
        between the server and the client is lost
        @param reason: Reason for loss of connection
        """
        print "Connection lost"
        print reason

    def lineReceived(self, data):
        """
        Overridden event handler for when a line of data is 
        received from client
        @param data: The data received from the client
        """
        print 'in lineReceived'
        print 'data => ' + data

class ServerFactory(Factory):
    protocol = LineServer

if __name__ == '__main__':
    factory = ServerFactory()
    reactor.listenTCP(PORT, factory)
    reactor.run()

```

所有驼峰式方法都是被覆盖的 twisted 方法。我只是把它们打出来，让它们在被调用时打印到 stdout。现在让我们制作一个客户端，它有一个我们可以重启几次的反应器:

```py

import time
import twisted.internet

from twisted.internet import reactor, protocol
from twisted.protocols.basic import LineOnlyReceiver

HOST = 'localhost'
PORT = 9000

class Client:
    """
    Client class wrapper
    """
    def __init__(self, new_user):
        self.new_user = new_user

        self.factory = MyClientFactory()

    def connect(self, server_address=HOST):
        """
        Connect to the server
        @param server_address: The server address
        """
        reactor.connectTCP(server_address, PORT, self.factory,
            timeout=30)

class MyProtocol(LineOnlyReceiver):

    def connectionMade(self):
        """
        Overridden event handler that is fired when a connection
        is made to the server
        """
        print "client connected!"
        self.run_long_running_process()

    def lineReceived(self, data):
        """
        Gets the data from the server
        @param data: The data received from the server
        """
        print "Received data: " + data

    def connectionLost(self, reason):
        """
        Connection lost event handler
        @param reason: The reason the client lost connection 
            with the server
        """
        print "Connection lost"

    def run_long_running_process(self):
        """
        Run the process
        """
        print 'running process'
        time.sleep(5)
        print "process finished!"
        self.transport.write('finished' + '\r\n')
        reactor.callLater(5, reactor.stop)

class MyClientFactory(protocol.ClientFactory):
    protocol = MyProtocol

if __name__ == '__main__':
    # run reactor multiple times
    tries = 3
    while tries:
        client = Client(new_user=True)
        client.connect('localhost')
        try:
            reactor.run()
            tries -= 1
            print "tries " + str(tries)
        except Exception, e:
            print e
            import sys
            del sys.modules['twisted.internet.reactor']
            from twisted.internet import reactor
            from twisted.internet import default
            default.install()

```

这里我们创建了一个简单的客户端，它也只接受文本行( **LineOnlyReceiver** )。重启反应堆的魔力在于代码末尾的 **while** 循环。实际上，我在 twisted 的 **reactor.py** 文件的异常处理程序中找到了代码，这给了我灵感。基本上我们正在做的是导入 Python 的 **sys** 模块。然后我们从 **sys.modules** 中删除反应器，这允许我们重新移植它并重新安装默认反应器。如果您在一个终端上运行服务器，在另一个终端上运行客户机，您会看到客户机重新连接了三次。

### 包扎

正如我在开始时提到的，我对 twisted 还很陌生。你应该做的不是重启反应器，而是在另一个线程中运行它。或者您可以使用它的一个延迟调用或延迟线程来绕过重启反应器的“需要”。坦白地说，本文中的方法在某些情况下甚至不起作用。实际上，我曾试图在一个用 **contextlib 的****context manager**decorator 修饰的函数中重启反应器，但这不知何故阻止了代码正确运行。不管怎样，我认为这是一种重新加载模块的有趣方式。