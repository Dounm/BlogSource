---
title: "ParameterServer入门和理解"
date: 2016-10-10T23:46:08+08:00
tags: [ParameterServer]
categories: [DistributedComputing]
toc: true
---

## 1. Parameter Server介绍
参数服务器是一个编程框架，用于方便分布式并行程序的编写，其中重点在于对大规模参数的分布式存储和协同的支持。
 
机器学习系统相比于其他系统而言，有一些自己的独特特点。例如：

- 迭代性：模型的更新并非一次完成，需要循环迭代多次
- 容错性：即使在每个循环中产生一些错误，模型最终仍能收敛
- 参数收敛的非均匀性：有些参数几轮迭代就会收敛，而有的参数却需要上百轮迭代。

而且工业界需要训练大型的机器学习模型，一些广泛应用的特定的模型在规模上有两个特点：

1. 参数很大，超过单个机器的容纳的能力（大型LR和神经网络）
2. 训练数据太大，需要并行提速（大数据）

因此在这种需求下，类似MapReduce的框架就不能满足需求了。
而设计一个上述系统时，我们需要解决很多问题。类似频繁访问修改模型参数所需要消耗的巨大带宽，以及如何提高并行度，减少同步等待造成的延迟等等。

而参数服务器即为解决这种需求提出的。ParameterServer适用于大规模深度学习系统，大规模Logistic Regression系统，大规模主题模型，大规模矩阵分解等依赖SGD或L-BFGS最优化的算法。

### 1.1 发展历史

参数服务器的概念最早来自于Alex Smola于2010年提出的并行LDA的框架。它通过采用一个分布式的Memcached作为存放参数的存储，这样就提供了有效的机制用于分布式系统中不同的Worker之间同步模型参数，而每个Worker只需要保存他计算时所以来的一小部分参数即可。

后来由Google的Jeff Dean进一步提出了第一代Google大脑的解决方案：**DistBelief**。DistBelief将巨大的深度学习模型分布存储在全局的参数服务器中，计算节点通过参数服务器进行信息传递，很好地解决了SGD和L-BFGS算法的分布式训练问题。
 
在后来就是李沐所在的DMLC组所设计的参数服务器。根据论文中所写，该parameter server属于第三代参数服务器，就是提供了更加通用的设计。架构上包括一个Server Group和若干个Worker Group。
![parameterserver_0](https://raw.githubusercontent.com/Dounm/TheFarmOfDounm/master/resources/images/parameter_server/1.png)

## 2 Parameter Server架构与实现
本文所针对的架构为李沐的
[Parameter Server for Distributed Machine Learning](https://www.cs.cmu.edu/~muli/file/ps.pdf)
和[Scaling Distributed Machine Learning with Parameter Server](https://www.cs.cmu.edu/~muli/file/parameter_server_osdi14.pdf)两篇论文的内容。

### 2.1 框架介绍

首先，该PS框架所假设的硬件情况是机器并不可靠，可能在训练中会重启，移动；数据有可能丢失；网络延迟同样也可能很高。因此，在这种情况下，对于那些使用*sychronous iterative communication pattern*的框架来说，如基于Hadoop的Mahout，就会在训练过程中由于机器的性能表现不均匀而变得非常的慢。
 
而对于Parameter Server来说，计算节点被分成两种：worker和servers。
workers保留一部分的训练数据，并且执行计算。而servers则共同维持全局共享的模型参数。而worker只和server有通信，互相之间没有通信。

parameter server具有以下特点：

- Efficient Communication：高效的通信。网络通信开销是机器学习分布式系统中的大头，因此parameter server基本尽了所有的努力来降低网络开销。
其中最重要的一点就是：异步通信。因为是异步通信，所以不需要停下来等一些慢的机器执行完一个iter，这就大大减少了延迟。
当然并非所有算法都天然的支持异步和随机性，有的算法引入异步后可能收敛会变慢，因此就需要自行在算法收敛和系统效率之间权衡。
- Elastic Scalability：使用一致性哈希算法，使得新的Server可以随时动态插入集合中，无需重新运行系统
- Fault Tolerance and Durability：
节点故障是不可避免的。对于server节点来说，使用链备份来应对；而对于Worker来说，因为worker之间互相不通信，因此在某个worker失败后，新的worker可以直接加入 
- Ease of Use：全局共享的参数可以被表示成各种形式：vector, matrices或是sparse类型，同时框架还提供对线性代数类型提供高性能的多线程计算库。

### 2.2 系统架构
![parameterserver_0](https://raw.githubusercontent.com/Dounm/TheFarmOfDounm/master/resources/images/parameter_server/1.png)
Paraeter Server框架中，每个server都只负责分到的部分参数（server共同维持一个全局共享参数）。server节点可以和其他server节点通信，每个server负责自己分到的参数，server group共同维持所有参数的更新。server manage node负责维护一些元数据的一致性，例如各个节点的状态，参数的分配情况。
 
worker节点之间没有通信，只和对应的server有通信。每个worker group有一个task scheduler，负责向worker分配任务，并且监控worker的运行情况。当有worker退出或加入时，task scheduler重新分配任务。

#### 2.2.1 (Key,Value) Vectors
参数都被认为是(key, value)集合。例如对于常见的LR来说，key就是feature ID，value就是其权值。对于不存在的key，可认为其权值为0。

大多数的已有的框架都是这么对(key, value)进行抽象的。但是PS框架除此之外还把这些(k,v)对看作是稀疏的线性代数对象（通过保证key是有序的情况下），因此在对vector进行计算操作的时候，也会在某些操作上使用BLAS库等线性代数库进行优化。

#### 2.2.2 Range Push/Pull

PS框架中，workers和servers之间通信是通过  push() 和 pull() 方法进行的。worker通过push()将计算好的梯度发送到server，然后通过pull()从server获取更新参数。

为了提高通信效率，PS允许用户使用Range Push/Range Pull操作。

    w.push(R, dest)
    w.pull(R, dest)

#### 2.2.3 Asynchronous Tasks and Dependency

task即为一个RPC调用函数。举例而言，worker对server的RPC调用`push()`或`pull()`都是算作一个task，又或是scheduler中用户自定义的一个用来对任意其他节点进行RPC调用的函数都算做是task。

而异步指的是，Tasks在被caller调用后立刻返回，但只有当caller收到callee的回复后，才将该task标记为结束。（在push()和pull()的例子中，worker==caller调用者，server==callee被调用者）
默认而言，callee为了性能执行tasks都是并行的，但是如果caller希望串行执行task的话，可以在不同的task之间添加*execute-after-finished*依赖。

举例：
![parameterserver_1](https://raw.githubusercontent.com/Dounm/TheFarmOfDounm/master/resources/images/parameter_server/3.png)

上图中，Task Scheduler的 调用了Worker中的`WORKERITERATE()`，这算做是一个task。其中Task Scheduler是caller，worker是callee。

而`WORKERITERATE()`这个task却包括3步，第一步是计算梯度，因为是本地的，所以不算task。但是后续的`push()`和`pull()`都是subtask。
 
假设，Scheduler要求iter10和iter11是独立的，但是iter11和iter12是添加了依赖的。那么对于其中某个worker来说，他的执行过程就类似于下图
![parameterserver_2](https://raw.githubusercontent.com/Dounm/TheFarmOfDounm/master/resources/images/parameter_server/4.png)

该worker在iter10的梯度计算后就立刻计算iter11的梯度，然而在iter11的梯度计算完后，却等待`push()`+`pull()`两个subtask结束才开始iter12的计算。

#### 2.2.4 Flexible Consistency
异步task能够提高系统的效率，但是却会造成数据不一致，从而降低算法的收敛速率。

上例中，该worker在iter10的结果参数$W_{11}$被`pull()`之前就开始计算iter11的梯度，因此他使用的模型参数仍然是$W_{10}$，所以iter11计算得到的梯度和iter10相同，这就拖慢了算法的收敛速率。

但有些算法对于数据的不一致性不那么敏感，因此这需要开发者自行权衡系统性能和算法收敛速率，这就需要考虑：

1. 算法对于参数非一致性的敏感度
2. 训练数据特征之间的关联度
3. 硬盘的存储容量

考虑到这个问题，PS为用户提供了多种任务依赖方式：
![parameterserver_3](https://raw.githubusercontent.com/Dounm/TheFarmOfDounm/master/resources/images/parameter_server/5.png)

1. **Sequential**：所有task一个接一个的执行，这个和单线程实现对等。（即**Bulk Synchronous Processing, BSP一致性约束**）
2. **Eventual**：与1正好相反，完全并行。这个仅推荐用于算法足够robust的情况
3. **Bounded Delay**：设置一个最大延时时间$\tau$。即只有与当前task时间差距大于$\tau$的tasks都被完成了，才能执行当前task。
因此，$\tau=0$，即Sequential；$\tau=\infty$，即Eventual

下面是使用**bounded delay**的PGD(proximal gradient descent)算法的系统运行流程：
![parameterserver_4](https://raw.githubusercontent.com/Dounm/TheFarmOfDounm/master/resources/images/parameter_server/6.png)

#### 2.2.5 User-defined Filters

PS所支持的另外一个减少网络带宽的方法是支持用户自定义过滤器来过滤掉那些比较小的被push的entry。

常用的过滤器有**significantly modified filter**，即只push大于某一门槛的entry。也有**KKT filter**，利用最优化问题的一些条件过滤掉对weights影响不大的entry。

### 2.3 parameter server的异步性与非凸问题

参数服务器模型更新的时候，worker的模型参数与server的模型参数可能有所不一致。
 
举例而言，梯度计算需要基于某个特定的参数值（相当于下山，我们只能找到针对特定某点的下山最快的方向，一旦该点变化，则下山最快的方向也就不对了）。问题在于：节点A从server获得参数值后，当计算完梯度后，此时server端的参数值可能已经被节点B所更新了。

但是在非凸问题（例如深度学习的优化）中，这反而是个好事，引入了**随机性**。这是因为非凸问题本身就不是梯度下降能够解决的，正常的单机迭代肯定会收敛到局部最优。有时我们常常会用一些额外的方法来跳出局部最优：

- 多组参数值初始化
- 模拟退火
- 随机梯度下降

而上面所说的PS框架正好利用异步性引入了随机性，有助于跳出局部最优。因此在Google的DistBelief框架中，提出了**Downpour SGD**算法，就是尽最大可能利用了这种随机性。

### 2.4 实现

#### 2.4.1 Vector Clock

PS使用vector clock来记录每个节点的参数，用来跟踪数据状态或避免数据重复发送。但假设有n个节点，m个参数，那么vector clock的空间复杂度就是O(nm)，无法承受。

幸运的是，parameter server在push和pull的时候，都是range-based，因此这个range里参数共享的是同一个时间戳，这就降低了复杂度（具体实现见下面ps-lite源码剖析）。

#### 2.4.2 Messages

一条message包括时间戳，和(k,v)对。但是由于机器学习问题频繁的参数访问，导致信息的压缩是必然的。
有两种优化来压缩网络带宽：

- **key的压缩**：因为训练数据在分配之后通常不会改变，因此worker没必要每次都发送相同的key，只需要在接受方第一次接收时缓存起来即可。后续只需发送value
- **用户自定义的过滤器**：有些参数更新并非对最终优化有价值，因此用户可以自定义过滤规则来过滤掉一些不必要的传送。例如对于梯度下降来说，很小的梯度值是低效的，可以忽略；同样当更新接近最优的时候，值也是低效的，可以忽略。（这个通过KKT条件来判断） 
 

#### 2.4.3 Replication and Consistency
PS在数据一致性方面，使用的是**一致性哈希算法**，然后每个节点备份其逆时针的k个节点的参数。

**一致性哈希算法**：即将数据按照某种hash算法映射到环上，然后将机器按照同样的hash算法映射到环上，将数据存储到环上顺时针最近的机器上。

![parameterserver_5](https://raw.githubusercontent.com/Dounm/TheFarmOfDounm/master/resources/images/parameter_server/7.png)
如图，k=3，则S2，S3和S4复制了S1的参数。

有两种方式保证主节点与备份节点之间的数据一致性。

1. 默认的复制方式：Chain Replication链备份（强一致性）![parameterserver_6](https://raw.githubusercontent.com/Dounm/TheFarmOfDounm/master/resources/images/parameter_server/8.png)
即worker0 push的更新x，在server0经过处理后变成了f(x)。只有在f(x)备份到了server1后，这次的push才算结束，woker0才会收到ack。

    该备份方式对于一些需要频繁更新参数的算法，可能造成难以承受的网络带宽开销。（相当于把网络带宽乘以k倍，k是备份的个数），因此parameter server框架也支持如下方法。
2. Replication after Aggregation：先聚合多个worker节点的更新，再备份![parameterserver_7](https://raw.githubusercontent.com/Dounm/TheFarmOfDounm/master/resources/images/parameter_server/9.png)
    先聚合多个worker节点的更新，然后再备份。只有数据被备份了之后，worker0和worker1才会收到ack。

    这种方法会造成worker收到的ack的时间被推迟，但是在宽松的一致性条件下（即woker无需等到ack就可以继续下一轮迭代）却无关紧要。

#### 2.4.4 Server Management
添加新server：

1. server manager给新节点分配key 
2. range。这个会导致其他server的key range更改新节点获取key range的data，并且将其备份到备份节点去
3. server manager广播节点的更改。

删除server：

server maneger通过心跳信号确定一个server死亡，然后就会该server的key range分配给多个节点。
 
#### 2.4.5 Worker Management
添加新woker：

1. task scheduler给新woker分配数据
2. 该新worker从别的wokers或是文件系统重新读取训练数据，然后从servers处pull下参数
3. task scheduler广播该变化，有可能造成其他worker释放部分train data

删除woker：

删除worker通常是直接不管该节点，这是因为丢失一部分训练数据通常并不会影响模型训练结果，但是恢复一个worker的话可能会占用较多资源。
当然用户也可以选择用新worker来替代丢失的worker。

## 3 ps-lite框架

ps-lite框架是DMLC组自行实现的parameter server通信框架，是DMLC其他项目的核心，例如其深度学习框架MXNET就依赖了ps-lite的实现。
源码剖析请看：[ps-lite框架源码剖析](https://www.zybuluo.com/Dounm/note/529299)

## 4 参考资料

- [Scaling Distributed Machine Learning with the Parameter Server](https://www.cs.cmu.edu/~muli/file/parameter_server_osdi14.pdf)
- [参数服务器——分布式机器学习的新杀器](http://chuansong.me/n/2161528)
- [【深度学习&分布式】Parameter Server 详解](http://blog.csdn.net/cyh_24/article/details/50545780)
- [Github: dmlc/ps-lite](https://github.com/dmlc/ps-lite)
- [ps-lite源码学习](http://longmenwaideyu.com/article/ps-lite_code)
- [知乎：最近比较火的parameter server是什么？](https://www.zhihu.com/question/26998075)
- [Large Scale Distributed Deep Networks](http://www.cs.toronto.edu/~ranzato/publications/DistBeliefNIPS2012_withAppendix.pdf)
 