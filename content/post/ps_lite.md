---
title: "ps-lite源码剖析"
date: 2016-10-11T23:46:08+08:00
tags: [ps-lite, ParameterServer]
categories: [DistributedComputing]
toc: true
---

## 1 ps-lite介绍
ps-lite框架是DMLC组自行实现的parameter server通信框架（注意仅是通信框架），是DMLC其他项目的核心，例如其深度学习框架MXNET就依赖了ps-lite的实现。

## 2 源码剖析
> 针对的源码的commit-id为：36b015f

### 2.1 重要类

#### 2.1.1 PostOffice
PostOffice是全局管理类。每个节点中都有且只有一个该类的对象。

主要用来配置当前node的一些信息，例如当前node是哪种类型(server,worker,scheduler)，nodeid是啥，以及worker/server 的rank 到 node id的转换。

Postoffice类利用**Singleton模式**来保证只有一个对象。

#### 2.1.2 Van
核心通信类，每个节点只有一个该对象，是Postoffice对象的成员。

Van类负责建立起节点之间的互相连接（例如Worker与Scheduler之间的连接），并且开启本地的receiving thread用来监听收到的message。

Van的子类为ZMQVan，即为用zmq库实现了连接的底层细节（zmq库是一个开源库，对socket进行了优良的封装）

#### 2.1.3 Customer
用于通信的对象。

每个Customer都与某个node id相绑定，代表当前节点发送到对应node id节点。

Customer对象维护request和response的状态，其中`tracker_`成员记录每个请求可能发送给了多少节点以及从多少个节点返回。
`tracker_` 下标为每个req标识的timestamp。
 
Customer也会启动一个receiving thread，而它接受到的消息来自于Van的receiving thread，即每个节点的Van对象收到message后，根据message的不同，推送到不同的customer对象中。

#### 2.1.4 SimpleApp

简单的通信类。每次通信发送int型的head和string型的body。

SimpleApp对象中包括一个Customer对象用来控制请求连接。

#### 2.1.5 KVWorker

继承自SimpleApp，包括如下方法： `Push()`,`Pull()`,`Wait()`。

`Push()`和`Pull()`最后都会调用`Send()`函数，`Send()`对KVPairs进行切分，因为每个Server只保留一部分参数，因此切分后的SlicedKVpairs就会被发送给不同的Server。

切分函数可以由用户自行重写，默认为`DefaultSlicer`，每个SlicedKVPairs被包装成Message对象，然后用`van::send()`发送。

#### 2.1.6 KVServer

继承自SimpleApp，包含如下方法：`Process()`和`Response()`。

`Process()`被注册到Customer对象中，当Customer对象的receiving thread接受到消息时，就调用`Process()`对数据进行处理。`Process()`内部的逻辑是调用 用户自行实现的一个`std::function`函数对象 对数据进行处理。

`Response()`就仅仅是向调用的worker发送信息

#### 2.1.7 KVPairs

拥有`keys`, `values`, `lens`等3个数组。lens和keys大小相等，表示每个key对应的value的个数。lens可为空，此时values被平分。

举例而言，
若keys=[1,5]，lens=[2,3]，那么values[0],values[1]就对应的是keys[0]，而values[2],values[3],values[5]对应的就是keys[1]。
而如果len为空，则`values.size()`必须是`keys.size()`（此处为2）的倍数，key[0]和key[1]各对应一半的values。

#### 2.1.8 SArray

shared array，用`std::shared_ptr`实现的数组，用于替代`std::vector`，避免数组的深拷贝。

### 2.2 消息处理流程

无论是worker节点还是server节点，在程序的最开始都会执行`Postoffice::start()`。`Postoffice::start()`会初始化节点信息，并且调用`Van::start()`。而`Van::start()`则会让当前节点与Scheduler节点相连，并且**启动一个本地线程recv thread**来持续监听收到的message。
 
worker和server都继承自SimpleApp类，所以都有一个customer对象。
customer对象本身也会**启动一个recv thread**，其中调用注册的`recv_handle_`函数对消息进行处理。

对于worker来说，其注册的`recv_handle_`是`KVWorker::Process()`函数。因为worker的recv thread接受到的消息主要是从server处pull下来的KV对，因此该`Process()`主要是接收message中的KV对；

而对于Server来说，其注册的`recv_handle_`是`KVServer::Process()`函数。因此server接受的是worker们push上来的KV对，需要对其进行处理，因此该`Process()`函数中调用的用户通过`KVServer::set_request_handle()`传入的函数对象。
 
每个customer对象都拥有一个`tracker_`(`std::vector<std::pair<int, int>>`类型)用来记录每个请求发送和返回的数量。
`tracker_`的下标即为请求的timestamp，`tracker_[t].first`是该请求发送给了多少节点，`tracker[t]_.second`是该请求收到了多少节点的回复。
`customer::Wait()`就是一直阻塞直到`tracker_[t].first == tracker[t].second`，用这个来控制同步异步依赖。
 
每当`Van`的recv thread收到一个message时，就会根据customer id的不同将message发给不同的customer的recv thread。
同时该message对应的请求（设为req）则`tracker_[req.timestamp].second++`。

### 2.3 实现细节

#### 2.3.1 位运算表示node和node group
因为有时请求要发送给多个节点，所以ps-lite用了一个map来存储**每个id对应的实际的node节点**。
 
其中id：1,2,4分别表示Scheduler, ServerGroup, WorkerGroup。
这样只需要将请求的目标节点的id 设为4，便意味着将该请求发送到所有的worker node。

除此之外，如果某worker想要向所有的server和scheduler同时发送请求，则只需要将目标node_id设为3即可。因为$3=2+1=KServerGroup+kScheduler$。

这正是为什么会选择1,2,4的原因，因为在二进制下：$1=001,2=010,4=100$。因此1-7内任意一个数字都代表的是Scheduler/ServerGroup/WorkerGroup的某一种组合。
 
1-7的id表示的是node group，而后续的id（即$8,9,10,\cdots$）则表示单个的node。
其中$8,10,12,\cdots$表示$worker0,worker1,worker2,\cdots$ （即$2n+8$）；$9,11,13,\cdots$表示$server0,server1,server2,\cdots$（即$2n+9$）
 
如此来说，对于每一个新节点，需要将其对应多个id上。例如对于worker2来说，需要将它与4,4+1,4+2,4+1+2,12这4个id相对应。
 
#### 2.3.2 发送的数据结构KVPairs将keys数组和value数组拆开
KVPairs的数据结构并非按照 `vector<pair<key, vector<values>>>`，而是按照`vector<key>`, `vector<values>`来组成。
这是因为，对于worker来说，它所拥有的部分数据集train data通常都是不变的，那这些数据集所引用的keys通常也是不变的。
这样的话，worker和server之间互相通信的时候，就可以不发送vector<keys>，仅发送vector<values>了，可以降低一部分网络带宽。

## 3 简单的例子

    #include <iostream>
    #include "ps/ps.h"
    using namespace std;
    using namespace ps;
    template <class Val>
    class KVServerDefaultHandle1 {      //functor，用与处理server收到的来自worker的请求
    public:
        // req_meta是存储该请求的一些元信息，即请求来自于哪个节点，发送给哪个节点等等
        // req_data即发送过来的数据
        // server即指向当前server对象的指针
        void operator() (const KVMeta& req_meta, const KVPairs<Val>& req_data, KVServer<Val>* server) {
            size_t n = req_data.keys.size();
            KVPairs<Val> res;
            if (req_meta.push) { //收到的是push请求
                CHECK_EQ(n, req_data.vals.size());
            } else {            //收到的是pull请求
                res.keys = req_data.keys;
                res.vals.resize(n);
            }
            for (size_t i = 0;i < n; ++i) {
                Key key = req_data.keys[i];
                if (req_meta.push) {    //push请求
                    store[key] += req_data.vals[i]; //此处的操作是将相同key的value相加
                } else {                    //pull请求
                    res.vals[i] = store[key];
                }
            }
            server->Response(req_meta, res);
        }
    private:
        std::unordered_map<Key, Val> store;
    };
    void StartServer() {
        if (!IsServer()) return;
        cout << "num of workers[" << NumWorkers() << "]" << endl;
        cout << "num of servers[" << NumServers() << "]" << endl;
        auto server = new KVServer<float>(0);
        server->set_request_handle(KVServerDefaultHandle1<float>());   //注册functor
        RegisterExitCallback([server](){ delete server; });
    }
    void RunWorker() {
        if (!IsWorker()) return;
        cout << "start Worker rank = " << MyRank() << endl;
        KVWorker<float> kv(0);
        // init
        int num = 10;
        vector<Key> keys(num);
        vector<float> vals(num);
        int rank = MyRank();
        for (int i = 0;i < num; ++i) {
            keys[i] = i;
            vals[i] = i+10;
        }
        // push
        int repeat = 1;
        vector<int> ts;
        for (int i = 0;i < repeat; ++i) {
            ts.push_back(kv.Push(keys, vals));  //kv.Push()返回的是该请求的timestamp
        }
        for (int t : ts) kv.Wait(t);
        // pull
        std::vector<float> rets;
        kv.Wait(kv.Pull(keys, &rets));
        for (size_t i = 0;i < rets.size(); ++i) {
            cout << MyRank() << " rets[" << i << "]: " << rets[i] << endl;
        }
        cout << endl;
    }
    int main(int argc, char* argv[]) {
        StartServer();
        Start();    //启动,Postoffice::start()
        RunWorker();
        Finalize(); //结束。每个节点都需要执行这个函数。
        return 0;
    }
    
运行结果（1个server，3个worker的情况）：
![ps-lite_0](https://raw.githubusercontent.com/Dounm/TheFarmOfDounm/master/resources/images/ps-lite/ps-lite_0.png)

## 4 延伸阅读
[parameter server入门和理解](https://www.zybuluo.com/Dounm/note/517675)