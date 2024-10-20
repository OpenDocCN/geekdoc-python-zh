# Python:运行 Ping、Traceroute 等等

> 原文：<https://www.blog.pythonlibrary.org/2010/06/05/python-running-ping-traceroute-and-more/>

去年，我需要想出一种方法来用 Python 获取以下信息:获取路由表，从 ping 一系列 IP 中捕获数据，运行 tracert 并获取有关安装的 NIC 的信息。这些都需要在 Windows 机器上完成，因为它是诊断脚本的一部分，试图找出为什么机器(通常是笔记本电脑)无法连接到我们的 VPN。我最终创建了一个 wxPython GUI，让用户可以轻松运行，但是这些脚本没有 wx 也可以很好地工作。让我们看看他们长什么样！

## 主脚本

首先，我们将看一下整个剧本，然后检查每一个重要的部分。如果你想使用下面的代码，你需要 [wxPython](http://www.wxpython.org) 和 [PyWin32](http://sourceforge.net/projects/pywin32/files/pywin32/) 包。

```py

import os
import subprocess
import sys
import time
import win32com.client
import win32net
import wx

filename = r"C:\logs\nic-diag.log"

class RedirectText:
    def __init__(self,aWxTextCtrl):
        self.out=aWxTextCtrl

        if not os.path.exists(r"C:\logs"):
            os.mkdir(r"C:\logs")
        self.filename = open(filename, "w")

    def write(self,string):
        self.out.WriteText(string)
        if self.filename.closed:
            pass
        else:
            self.filename.write(string)

class MyForm(wx.Frame):

    #---------------------------------------------------------------------- 
    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY, "Diagnostic Tool")

        # Add a panel so it looks the correct on all platforms
        panel = wx.Panel(self, wx.ID_ANY)
        log = wx.TextCtrl(panel, wx.ID_ANY, size=(300,100),
                          style = wx.TE_MULTILINE|wx.TE_READONLY|wx.HSCROLL)
        # log.Disable()
        btn = wx.Button(panel, wx.ID_ANY, 'Run Diagnostics')
        self.Bind(wx.EVT_BUTTON, self.onRun, btn)

        # Add widgets to a sizer        
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(log, 1, wx.ALL|wx.EXPAND, 5)
        sizer.Add(btn, 0, wx.ALL|wx.CENTER, 5)
        panel.SetSizer(sizer)

        # redirect text here
        self.redir=RedirectText(log)
        sys.stdout=self.redir

    #----------------------------------------------------------------------    
    def runDiagnostics(self):
        """
        Run some diagnostics to get the machine name, ip address, mac,
        gateway, DNS, route tables, etc
        """
        # create the route table:
        # based on the following list comp from http://win32com.goermezer.de/content/view/220/284/
        # route_table = [elem.strip().split() for elem in os.popen("route print").read().split("Metric\n")[1].split("\n") if re.match("^[0-9]", elem.strip())]
        route_table = []
        proc = subprocess.Popen("route print", shell=True,
                        stdout=subprocess.PIPE)
        while True:
            line = proc.stdout.readline()
            route_table.append(line.strip().split())
            if not line: break
        proc.wait()

        print "Log Created at %s" % time.ctime()
        print "----------------------------------------------------------------------------------------------"
        info = win32net.NetWkstaGetInfo(None, 102)
        self.compname = info["computername"]
        print "Computer name: %s\n" % self.compname

        print "----------------------------------------------------------------------------------------------"
        print "Route Table:"
        print "%20s\t %15s\t %15s\t %15s\t %s" % ("Network Destination", "Netmask",
                                          "Gateway", "Interface", "Metric")
        for route in route_table:
            if len(route) == 5:
                dst, mask, gateway, interface, metric = route
                print "%20s\t %15s\t %15s\t %15s\t %s" % (dst, mask, gateway, interface, metric)

        print "----------------------------------------------------------------------------------------------\n"
        ips = ["65.55.17.26", "67.205.46.185", "67.195.160.76"]
        for ip in ips:
            self.pingIP(ip)
            print
            self.tracertIP(ip)
            print "\n----------------------------------------------------------"
        self.getNICInfo()
        print "############ END OF LOG ############"

    #----------------------------------------------------------------------   
    def pingIP(self, ip):
        proc = subprocess.Popen("ping %s" % ip, shell=True, 
                                stdout=subprocess.PIPE) 
        print
        while True:
            line = proc.stdout.readline()                        
            wx.Yield()
            if line.strip() == "":
                pass
            else:
                print line.strip()
            if not line: break
        proc.wait()

    #----------------------------------------------------------------------
    def tracertIP(self, ip):
        proc = subprocess.Popen("tracert -d %s" % ip, shell=True, 
                                stdout=subprocess.PIPE)
        print 
        while True:
            line = proc.stdout.readline()
            wx.Yield()
            if line.strip() == "":
                pass
            else:
                print line.strip()
            if not line: break
        proc.wait()

    #----------------------------------------------------------------------            
    def getNICInfo(self):
        """
        http://www.microsoft.com/technet/scriptcenter/scripts/python/pyindex.mspx?mfr=true
        """
        print "\nInterface information:\n"
        strComputer = "."
        objWMIService = win32com.client.Dispatch("WbemScripting.SWbemLocator")
        objSWbemServices = objWMIService.ConnectServer(strComputer,"root\cimv2")
        colItems = objSWbemServices.ExecQuery("Select * from Win32_NetworkAdapterConfiguration")
        numOfNics = len(colItems)
        count = 1
        for objItem in colItems:
            # if the IP interface is enabled, grab its info
            print "***Interface %s of %s***" % (count, numOfNics)
            if objItem.IPEnabled == True:                
                print "Arp Always Source Route: ", objItem.ArpAlwaysSourceRoute
                print "Arp Use EtherSNAP: ", objItem.ArpUseEtherSNAP
                print "Caption: ", objItem.Caption
                print "Database Path: ", objItem.DatabasePath
                print "Dead GW Detect Enabled: ", objItem.DeadGWDetectEnabled
                z = objItem.DefaultIPGateway
                if z is None:
                    a = 1
                else:
                    for x in z:
                        print "Default IP Gateway: ", x
                print "Default TOS: ", objItem.DefaultTOS
                print "Default TTL: ", objItem.DefaultTTL
                print "Description: ", objItem.Description
                print "DHCP Enabled: ", objItem.DHCPEnabled
                print "DHCP Lease Expires: ", objItem.DHCPLeaseExpires
                print "DHCP Lease Obtained: ", objItem.DHCPLeaseObtained
                print "DHCP Server: ", objItem.DHCPServer
                print "DNS Domain: ", objItem.DNSDomain
                z = objItem.DNSDomainSuffixSearchOrder
                if z is None:
                    a = 1
                else:
                    for x in z:
                        print "DNS Domain Suffix Search Order: ", x
                print "DNS Enabled For WINS Resolution: ", objItem.DNSEnabledForWINSResolution
                print "DNS Host Name: ", objItem.DNSHostName
                z = objItem.DNSServerSearchOrder
                if z is None:
                    a = 1
                else:
                    for x in z:
                        print "DNS Server Search Order: ", x
                print "Domain DNS Registration Enabled: ", objItem.DomainDNSRegistrationEnabled
                print "Forward Buffer Memory: ", objItem.ForwardBufferMemory
                print "Full DNS Registration Enabled: ", objItem.FullDNSRegistrationEnabled
                z = objItem.GatewayCostMetric
                if z is None:
                    a = 1
                else:
                    for x in z:
                        print "Gateway Cost Metric: ", x
                print "IGMP Level: ", objItem.IGMPLevel
                print "Index: ", objItem.Index
                z = objItem.IPAddress
                if z is None:
                    a = 1
                else:
                    for x in z:
                        print "IP Address: ", x
                print "IP Connection Metric: ", objItem.IPConnectionMetric
                print "IP Enabled: ", objItem.IPEnabled
                print "IP Filter Security Enabled: ", objItem.IPFilterSecurityEnabled
                print "IP Port Security Enabled: ", objItem.IPPortSecurityEnabled
                z = objItem.IPSecPermitIPProtocols
                if z is None:
                    a = 1
                else:
                    for x in z:
                        print "IP Sec Permit IP Protocols: ", x
                z = objItem.IPSecPermitTCPPorts
                if z is None:
                    a = 1
                else:
                    for x in z:
                        print "IP Sec Permit TCP Ports: ", x
                z = objItem.IPSecPermitUDPPorts
                if z is None:
                    a = 1
                else:
                    for x in z:
                        print "IPSec Permit UDP Ports: ", x
                z = objItem.IPSubnet
                if z is None:
                    a = 1
                else:
                    for x in z:
                        print "IP Subnet: ", x
                print "IP Use Zero Broadcast: ", objItem.IPUseZeroBroadcast
                print "IPX Address: ", objItem.IPXAddress
                print "IPX Enabled: ", objItem.IPXEnabled
                z = objItem.IPXFrameType
                if z is None:
                    a = 1
                else:
                    for x in z:
                        print "IPX Frame Type: ", x
                print "IPX Media Type: ", objItem.IPXMediaType
                z = objItem.IPXNetworkNumber
                if z is None:
                    a = 1
                else:
                    for x in z:
                        print "IPX Network Number: ", x
                print "IPX Virtual Net Number: ", objItem.IPXVirtualNetNumber
                print "Keep Alive Interval: ", objItem.KeepAliveInterval
                print "Keep Alive Time: ", objItem.KeepAliveTime
                print "MAC Address: ", objItem.MACAddress
                print "MTU: ", objItem.MTU
                print "Num Forward Packets: ", objItem.NumForwardPackets
                print "PMTUBH Detect Enabled: ", objItem.PMTUBHDetectEnabled
                print "PMTU Discovery Enabled: ", objItem.PMTUDiscoveryEnabled
                print "Service Name: ", objItem.ServiceName
                print "Setting ID: ", objItem.SettingID
                print "Tcpip Netbios Options: ", objItem.TcpipNetbiosOptions
                print "Tcp Max Connect Retransmissions: ", objItem.TcpMaxConnectRetransmissions
                print "Tcp Max Data Retransmissions: ", objItem.TcpMaxDataRetransmissions
                print "Tcp Num Connections: ", objItem.TcpNumConnections
                print "Tcp Use RFC1122 Urgent Pointer: ", objItem.TcpUseRFC1122UrgentPointer
                print "Tcp Window Size: ", objItem.TcpWindowSize
                print "WINS Enable LMHosts Lookup: ", objItem.WINSEnableLMHostsLookup
                print "WINS Host Lookup File: ", objItem.WINSHostLookupFile
                print "WINS Primary Server: ", objItem.WINSPrimaryServer
                print "WINS Scope ID: ", objItem.WINSScopeID
                print "WINS Secondary Server: ", objItem.WINSSecondaryServer
                print "-------------------------------------------------------\n"
            else:
                print "Interface is disabled!\n"
            count += 1

    #----------------------------------------------------------------------
    def onRun(self, event):
        self.runDiagnostics()
        self.redir.filename.close()
        # Restore stdout to normal
        sys.stdout = sys.__stdout__

#----------------------------------------------------------------------         
# Run the program
if __name__ == "__main__":
    app = wx.PySimpleApp()
    frame = MyForm().Show()
    app.MainLoop()

```

与大多数 Python 程序一样，这个程序从各种导入开始。接下来我们创建一个简单的类(RedirectText ),我们将使用它来帮助我们将 stdout 重定向到 wx。TextCtrl 和一个日志文件。这通过传入 wx 的实例来实现。TextCtrl，然后设置“sys.stdout”以指向它(请参见 MyForm 类中的 __init__ 方法)。在 *RedirectText* 类之后，我们有了 *MyForm* 类，在那里我们创建了 wxPython GUI。实际上 GUI 本身并没有太多的东西。只是一个多行文本控件和一个面板上的按钮，但这就是我们所需要的。这个类的其余部分由收集所有我们需要的信息并将其记录到屏幕和文件中的方法组成。

现在就来看看那些方法吧！注意，这些方法是从 *runDiagnostics* 方法调用的，该方法是从 *onRun* 按钮事件处理程序启动的。

## 获取路由表(又名:IP 路由)

当我研究如何做到这一点时，我在另一个[博客](http://win32com.goermezer.de/content/view/220/284/)上发现了以下脚本:

```py

import os, re
route_table = [elem.strip().split() for elem in os.popen("route print").read().split("Metric\n")[1].split("\n") if re.match("^[0-9]", elem.strip())]

```

我发现这很难理解，所以我重新写了一遍(或者找到了另一个例子，但忘了记下来),如下所示:

```py

route_table = []
proc = subprocess.Popen("route print", shell=True,
                stdout=subprocess.PIPE)
while True:
    line = proc.stdout.readline()
    route_table.append(line.strip().split())
    if not line: break
proc.wait()

print "----------------------------------------------------------------------------------------------"
print "Route Table:"
print "%20s\t %15s\t %15s\t %15s\t %s" % ("Network Destination", "Netmask",
                                  "Gateway", "Interface", "Metric")
for route in route_table:
    if len(route) == 5:
        dst, mask, gateway, interface, metric = route
        print "%20s\t %15s\t %15s\t %15s\t %s" % (dst, mask, gateway, interface, metric)

```

我发现上面的代码更容易阅读和理解。它所做的只是使用子流程模块运行“route print”并将结果写入 stdout。不要被上面的 *proc.stdout* 迷惑了。这是进程的标准输出，不是普通的标准输出。我们希望将数据重定向到普通的标准输出！为此，我们读取 proc 的 stdout(或者有人会说是管道),并将每一行数据追加到一个列表中。然后，我们使用 Python 的字符串格式创建一个很好的定制输出。现在我们来看看如何使用 Python 运行 Ping 和 Tracert。

## 使用 Python 运行 Ping / Tracert

使用 Python 执行 ping 操作非常简单。我们只需要子流程模块来完成这项工作，正如您在下面的代码片段中看到的:

```py

def pingIP(self, ip):
    proc = subprocess.Popen("ping %s" % ip, shell=True, 
                            stdout=subprocess.PIPE) 
    print
    while True:
        line = proc.stdout.readline()                        
        wx.Yield()
        if line.strip() == "":
            pass
        else:
            print line.strip()
        if not line: break
    proc.wait()

```

在这段代码中，我们使用 wx。实时将 ping 结果发送到我们的文本控件。如果我们没有这样做，那么在 ping 完成运行之前，我们不会收到任何 ping 结果。注意，我们也使用无限循环来获取结果。一旦结果不再出现，我们就跳出这个循环。如果您查看 tracert 代码，您会发现唯一的区别在于 out *子流程。Popen* 命令。这将是重构的一个很好的候选，但是我将把它作为一个练习留给读者。

## 使用 Python 获取网卡信息

微软在他们的 [Technet](http://www.microsoft.com/technet/scriptcenter/scripts/python/pyindex.mspx?mfr=true) 子网站上有一整套 Python 脚本，我最终使用它们来获取我们电脑中网络接口卡(NIC)的各种有用信息。我不会在这里重复代码，因为它很长，我们已经有了。然而，这很容易理解，我怀疑如果你知道自己在做什么，你可以通过 WMI 获得同样的信息。我们感兴趣的主要部分是 MAC 和 IP 地址。让我们从长代码中提取这些信息，看看它有多容易获得:

```py

strComputer = "."
objWMIService = win32com.client.Dispatch("WbemScripting.SWbemLocator")
objSWbemServices = objWMIService.ConnectServer(strComputer,"root\cimv2")
colItems = objSWbemServices.ExecQuery("Select * from Win32_NetworkAdapterConfiguration")
numOfNics = len(colItems)

for objItem in colItems:
    z = objItem.IPAddress
    if z is None:
        a = 1
    else:
        for x in z:
             print "IP Address: ", x
    print "MAC Address: ", objItem.MACAddress

```

很简单，是吧？快看。您使用类似 SQL 的语法来运行查询。这就是为什么我认为你可以使用 WMI(事实上，这可能是它在以一种迟钝的方式做)。不管怎样，这就是全部了。

## 包扎

现在，您知道了从 PC 获取各种网络信息的秘密，以及如何将子进程的管道重定向到日志文件和 wxPython 文本控件。你如何选择使用这些信息取决于你自己。