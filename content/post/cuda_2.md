---
title: "CUDA程序调优指南（三）：BlockNum和ThreadNumPerBlock"
date: 2018-12-12T23:48:08+08:00
tags: [CUDA]
categories: [CUDA]
toc: true
---

**（以下纯属经验而谈，并非一定准确）**

## x. ThreadNumPerBlock

对于`ThreadNumPerBlock`而言，其上限由硬件限制，有两个因素

- 一个是MaxThreadsPerBlock
- 一个是$\frac{MaxRegisterPerBlock}{RegisterPerThread}$。写好了Kernel后，其RegisterPerThread是固定值。该值由编译器确定，可由nvcc的`--ptxas-options=-v`得出。


`ThreadNumPerBlock`通常取值是256/512/1024（经验而谈，值越大越好）。 

但有时预先选好的值达不到100% **Occupancy**，所以选取可以达到最高Occupancy的最大值。

那么，什么损失Occupancy？

### x.1 Occupancy的定义

> a CUDA device's hardware implementation groups adjacent threads within a block into warps.
A warp is considered active from the time its threads begin executing to the time when all threads in the warp have exited from the kernel.

**Occupancy：一个SM上active warp 比上 该SM最大的active warps的数量的比值**。

Low Occupancy会导致较低的instruction issue effiency（参考1.4节所说的关于**latency**的定义），因为没有足够多的可用warp来掩盖互相依赖的instruction之间的延迟。所以我们需要尽可能让Occupancy更大。



Occupancy分为两种【Theoretical Occupancy】和【Achieved Occupancy】。Achieved Occupancy受制于Theoretical Occupancy。 

### x.2. Theoretical Occupancy，ThreadsPerBlock与RegisterPerThread

首先，如何根据`ThreadsPerBlock`和`RegisterPerThread`计算Theoretical Occupancy？

1. 假设预先设置`ThreadsPerBlock`，可以得到`WarpPerBlock`
2. 计算 $BlocksPerSM = \frac{RegisterPerSM}{RegisterPerThread*ThreadsPerBlock}$（注意整数相除，下取整）
3. 计算 $WarpsPerSM=WarpsPerBlock*BlocksPerSM$，对比该值与`MaxWarpsPerSM`，是否达到100%。

上述计算中，`RegisterPerSM`和`RegisterPerThread`都是常量。

所以如未达到100%，则可以尝试更改`ThreadsPerBlock`看是否能达到更高Occupancy。

 

> 举例：
>
> 以1080（CC 6.1）为例，其RegisterPerSM是65536，MaxWarpsPerSM=64。
>
> 针对于某个实现的Im2Col kernel function而言，其RegisterPerThread是39。
>
> 那么，设置ThreadsPerBlock=1024时，warpsPerBlock=1024/32=32，BlocksPerSM=1。
>
> WarpsPerSM=32*1=32，则Theoretical Occupancy仅为50%，对应执行时间为86us。
>
> - 若ThreadsPerBlock=512，那么Theoretical Occupancy=75%，执行时间减少为76us。
>
> - 若ThreadsPerBlock=768，那么Theoretical Occupancy=75%，执行时间更少，为64us。

 

### x.3. Achieved Occupancy

Achieved Occupancy无法高于Theoretical Occupancy，但有时会达不到理论值，具体如何见[Achieved Occupancy](https://docs.nvidia.com/gameworks/content/developertools/desktop/analysis/report/cudaexperiments/kernellevel/achievedoccupancy.htm)。

## xx. BlockNum

`BlocksNum`的取值则更有讲究，$BlocksNum=BlocksPerSM * NumofSM$，我们只需要求解`BlocksPerSM`即可。



因为GPU执行机制的原因，理论上`BlocksPerSM`可以很大。因为如果每个SM平均很多Blocks，但SM每次只能并发执行两个Block，那后面的Block会放到stream里等到前面的Block执行完毕才能被SM执行。

但通常来说，在占满SM资源的情况下，BlocksPerSM越小越好。

> 结合`CUDA_1D_LOOP`来看，BlocksPerSM越小，总的Block数量就少，每个thread所处理的任务量多，可以减少一些创建Block的资源开销，如shared memory的初始化。

 

针对于一个SM最大可以【并发concurrently】执行多少个Block，有如下几个因素限制上限：

- $MaxBlocksPerSM$：硬件属性
- $\frac{MaxThreadsPerSM}{ThreadsPerBlock}$
- $\frac{SharedMemPerSM}{SharedMemPerBlock}$
- $\frac{RegisterPerSM}{RegisterPerThread * ThreadsPerBlock}$

因此，我们取这三个值的最小值作为`BlocksPerSM`即可。