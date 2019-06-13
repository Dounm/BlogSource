# GBDT算法详解（二）：GBDT与Xgboost

系列文章分为两部分：

1. GBDT算法详解（一）：决策树与Boosting算法
2. GBDT算法详解（二）：GBDT与Xgboost

---

[TOC]


## 1. GBDT（Gradient Boosting Decision Tree）和 Xgboost

GBDT，Gradient Boosting Descision Tree，就是前面所提到的Gradient Boosting与Boosting Tree结合的结果。GBDT所采用的也是**加法模型**和**前向分步算法**，树的类型则是**CART树**，loss函数不定。

### 1.1 GBDT的目标函数

对于普通的机器学习模型而言，其目标函数可以定义为如下：
$$
Obj = \sum^n_{i=1}l(y_i,\hat{y}_i) + \sum^K_{k=1}\Omega(f_k)
$$
其中，前面的是$loss$函数，后面的$\Omega$是正则化项。

结合前述的前向分步算法的原理，在第$t$步时，目标函数就是
$$
Obj^{(t)}=\sum^n_{i=1}l(y_i, \hat{y}^t_i) +\sum^t_{i=1}\Omega(f_i) \\
= \sum^n_{i=1}l(y_i,\hat{y}^{t-1}_i+f_t(x_i))+\Omega(f_t)+constant \\
\label{obj}
$$
此时最优化该目标函数，就求得了$f_t{x_i}$。

#### 1.1.2 负梯度的理论支撑

前面第3.3节提到Gradient Boosting时，提及Gradient Boosting以负梯度代替残差来求解基函数，实际上，负梯度的理论支撑则是**泰勒公式的一阶展开**。即
$$
f(x+\Delta x)\approx f(x)+f'(x)\Delta x
$$
对$l(y_i,\hat{y}^{t-1}_i+f_t(x_i))$作泰勒一阶展开，得到
$$
l(y_i,\hat{y}^{t-1}_i+f_t(x_i))=l(y_i,\hat{y}^{t-1}_i)+g_if_t(x_i) \\
g_i是l(y_i,\hat{y}^{t-1}_i)关于\hat{y}^{t-1}的一阶导
$$
此时公式$\ref{obj}$目标函数（不考虑正则项）则变成
$$
Obj^{(t)} = \sum^n_{i=1}[l(y_i,\hat{y}^{t-1}_i)+g_if_t(x_i)] \\
Obj^{(t-1)} = \sum^n_{i=1}[l(y_i,\hat{y}^{t-1}_i)]
$$
我们肯定希望$Obj​$函数每步都减小的，即$Obj^{(t)} < Obj^{(t-1)}​$，那么关键就在于$g_if_t(x_i)​$这一项了。因为我们不知道$g_i​$到底是正还是负，那么只需让$f_t(x_i)=-\alpha g_i​$（$\alpha​$是我们任取的一个正系数）就能让$g_if_t(x_i)​$一直恒为负了。



#### 1.1.2 xgboost的目标函数

xgboost是GBDT的一个变种，也是最广泛的一种应用。它在对$loss$函数进行泰勒展开时，取的是**二阶展开**而非一阶展开，因此与实际$Obj$函数的值更接近（前提条件要求$loss$函数二阶可导）。

泰勒公式的二阶展开为：
$$
f(x+\Delta x)\approx f(x)+f'(x)\Delta x+\frac{1}{2}f''(x)\Delta x^2
$$


带入到公式$\ref{obj}$中，则
$$
Obj^{(t)} \approx \sum^n_{i=1}[l(y_i,\hat{y}^{t-1}_i)+g_if_t(x_i)+\frac{1}{2}h_if_t^2(x_i)]+\Omega(f_t) \\
g_i是l(y_i,\hat{y}^{t-1}_i)关于\hat{y}^{t-1}的一阶导 \\
h_i是l(y_i,\hat{y}^{t-1}_i)关于\hat{y}^{t-1}的二阶导
$$
因为在第$t$步时，$\hat{y}^{t-1}_i$是已知值，所以$l(y_i,\hat{y}^{t-1}_i)$是常数，不影响函数优化，可以省去，则

$$
Obj^{(t)} \approx \sum^n_{i=1}[g_if_t(x_i)+\frac{1}{2}h_if_t^2(x_i)]+\Omega(f_t)
\label{xgboost_loss}
$$
对这个Obj进行优化得到的就是第$t$步的$f_t(x)$，然后最终将每一步的$f(x)$加在一起就得到了整体模型。

> 后续章节中，我们都使用xgboost的目标函数来进行推导。


### 1.2 用决策树来表示$f_t(x)$和$Obj^{(t)}$

对于决策树来说，设它有$T$个叶子结点（即第二节提到的路径单元cell），那么每个叶子结点都有固定的值$w$，且每个样本都必然存在且仅存在于一个叶子节点上，因此
$$f_t(x)=w_{q(x)} \\
q(x)代表样本x位于哪个叶子结点\\
w_q代表该叶子结点的取值$$

另一方面，决策树的复杂度可以由$\Omega(f_t)=\gamma T+\frac{1}{2}\lambda\sum^T_{j=1}w_j^2$来定义，即决策树叶子结点的数量和叶子结点取值的L2范数。

因此，假设$
I_j=\{i|q(x_i)=j\}$为第$j$个叶子结点的样本集合，那么公式$\ref{xgboost_loss}$就变为如下形式：
$$
Obj^{(t)} \approx \sum^n_{i=1}[g_if_t(x_i)+\frac{1}{2}h_if_t^2(x_i)]+\Omega(f_t)\\
=\sum^n_{i=1}[g_iw_{q(x_i)}+\frac{1}{2}h_iw^2_{q(x_i)}]+\gamma T+\frac{1}{2}\lambda\sum^T_{j=1}w_j^2 \\
= \sum^T_{j=1}[(\sum_{i\in I_j}g_i)w_j+\frac{1}{2}(\sum_{i\in I_j}h_i+\lambda)w^2_j]+\gamma T \\
令G_j=\sum_{i\in I_j}g_i，H_j=\sum_{i\in I_j}h_i \\
= \sum^T_{j=1}[G_jw_j+\frac{1}{2}(H_j+\lambda)w_j^2]+\gamma T
$$
当位于第$t$步时，在树的结构固定的情况下，我们已经知道每个叶子结点有哪些样本，则$q$和$I_j$是确定的。又因为$g_i$和$h_i$是$t-1$步的导数，所以也是固定的，因此$G_j$和$H_j$都是固定的。

令$Obj$函数关于$w_j$的一阶导为$0$，即可求得最优的$w_j$（$obj$函数对于$w_j$来说，是个凸函数），即
$$
w_j^*=-\frac{G_j}{H_j+\lambda}
$$

> 注意，二阶泰勒展开的$obj$函数针对于$w_j$是凸函数，因此可以直接求出最优解析解。而一阶展开时的$obj$函数不是可导的，所以只能逐步降低$obj$函数。（估计这也是XGBoost选择二阶泰勒展开的原因之一）

则针对于结构固定的决策树，最优的$obj$函数即为：
$$
Obj=-\frac{1}{2}\sum^T_{j=1}\frac{G_j^2}{H_j+\lambda}+\gamma T
$$

### 1.3 求解最优结构的决策树

前面提到，对于固定结构的决策树，我们可以得知其最优的$obj$函数的值。那么，该如何求解最优结构的决策树呢？

第2.1节提到，决策树通常采用后剪枝的方案，因此其学习分为两个阶段：决策树的生成（特征选择在生成阶段处理）和决策树的剪枝。其中决策树的生成阶段仅考虑如何更好的对训练数据进行拟合，只有到了决策树的剪枝阶段才考虑到了减少模型复杂度的问题。

在普通的GBDT采用决策树是CART，因此也是用后剪枝来处理过拟合的问题。

但是在xgboost中用的却是预剪枝。在$obj$函数中，我们既考虑了用$loss$拟合训练数据，又考虑了用正则项$\Omega$来减少模型复杂度。因此xgboost的在决策树的生成阶段就处理了过拟合的问题，无需独立的剪枝阶段。

具体步骤如下：

1. 从深度为0的书开始，对每个叶节点枚举所有的可用特征
2. 针对每个特征，把属于该节点的训练样本的该特征值升序排列，通过线性扫描的方式来决定该特征的最佳分裂点，并采用最佳分裂点时的收益
3. 选择收益最大的特征作为分裂特征，用该特征的最佳分裂点作为分裂位置，把该节点生成出左右两个新的叶子结点，并为每个新节点关联新的样本集
4. 回到第一步，继续递归直到满足特定条件。

那么如何计算上述收益呢？因为我们在某个节点二分成两个节点，分别是左L右R。因为除了当前待处理的节点，其他节点对应的$obj$中的值都不变，所以我们只需要考虑当前节点的$obj$值即可。

分类前的针对于该子节点的最优目标函数就是$Obj=-\frac{1}{2}\frac{(G_L+G_R)^2}{(H_L+H_R)+\lambda}+\gamma$，分裂后则变成了$Obj=-\frac{1}{2}[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}]+2\gamma$，那么对于该目标函数来说，分裂后的收益为：
$$Gain=\frac{1}{2}[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{(H_L+H_R)+\lambda}]-\gamma$$

于是便可以用上式来决定最优分裂特征和最优特征分割点。

### 1.4 总结

所以xgboost的算法为：

1. 算法在前向分步算法的每一步都新生成一颗决策树
2. 拟合这棵树之前，计算损失函数在每个样本上的一阶导和二阶导，即$g_i$和$h_i$
3. 通过贪心策略生成一棵树，计算每个叶子结点的$G_j$和$H_j$，计算预测值$w$
4. 把新生成的决策树$f_t(x)$加入$\hat{y}^t_i=\hat{y}^{t-1}_i+\epsilon f_t(x_i)$，其中$\epsilon$是学习率，为了抑制模型的过拟合

## 2. XGBoost的优化

XGBoost是DMLC开源在Github的Gradient Boosting框架，主要作者是陈天奇。它支持C++/Java/Python/R等语言，同样也支持Hadoop/Spark/Flink等分布式处理框架，且在数据竞赛中拥有优异的表现。

相比于普通的GBDT，XGBoost主要优点在于：
- 不仅支持决策树作为基分类器，还支持线性分类器
- 用到了$loss$函数的二阶泰勒展开，因此与损失函数更接近，收敛更快
- 在代价函数中加入了正则项，用于控制模型复杂度。正则项里包括了树的叶子节点个数和叶子结点输出值的L2范数，可以防止模型过拟合。
- Shrinkage，就是前面所述的$\epsilon$，主要用于削弱每科树的影响，让后面有更大的学习空间。实际应用中，一般把$\epsilon$设小点，迭代次数设大点。
- 列抽样（column sampling）。xgboost从随机森林算法中借鉴来的，支持列抽样可以降低过拟合，并且减少计算。
- 支持对缺失值的处理。对于特征值缺失的样本，xgboost可以学习这些缺失值的分裂方向。
- 支持并行。在对每颗树进行节点分裂时，需要计算每个特征的增益，选择最大的那个特征作为分裂特征，各个特征的增益计算可以开多线程进行
- 近似算法（Approximate Algorithm），树节点在分裂时，需要枚举每个可能的分割点。当数据没法一次性载入内存时，这种方法会很慢。xgboost提出了一种近似的方法去高效的生成候选分割点。

接下来选取几个比较重要的点详细讲讲

### 2.1 Approximate Split Finding Algorithm

树学习的关键问题就是找到最优分割点。常见的方法是枚举所有可能的分割点，称之为 exact greedy algorithm。

![4](https://raw.githubusercontent.com/Dounm/TheFarmOfDounm/master/resources/images/gdbt/4.png)

exact greedy algorithm计算量过大，而且当数据量较大没法全填入内存时，会很慢。因此xgboost引入了 approximate算法。

该算法对于某个特征$X_k$，首先通过特征分布来确定若干值域分界点$\{s_{k1},s_{k2},\dots,s_{kl}\}$。然后根据这些值域分界点把样本分入桶中，对每个桶内的样本统计值$G,H$进行累加，记为分界点的统计量。最后在分界点集合上进行贪心查找，得到的结果就是最佳分裂点的近似。

![5](https://raw.githubusercontent.com/Dounm/TheFarmOfDounm/master/resources/images/gdbt/5.png)

那么该如何寻找值域分界点$\{s_{k1},s_{k2},\dots,s_{kl}\}$呢？XGBoost中介绍了一种方法，叫**加权分位数略图 Weighted Quantile Sketch**。

为了尽可能的逼近最佳分裂点，我们需要保证采样后的数据分布和原始数据分布尽可能一直。令$D_k=\{(x_{1k},h_1),(x_{2k},h_2),\dots,(x_{nk},h_n)\}$表示每个训练样本的第$k$维特征的值和对应的二阶导数。然后定义排序函数为
$$
r_k(z)=\frac{\sum_{(x,h)\in D_k, x<z}h}{\sum_{(x,h)\in D_k}h} \\ 
即函数特征值小于z的样本分布占比，二阶导h是权重
$$

> 为什么使用二阶导h作为权重，参见Reference 6的附录部分。从公式上来看，当$h$越大时，$s_{k,j}$就越密集。

然后找到一组点$\{s_{k1},s_{k2},\dots,s_{kl}\}$满足：
$$
|r_k(s_{k,j})-r_k(s_{k,j+1})|<\varepsilon
$$
其中，$s_{k1}=min_i\ x_{ik}, s_{kl}=max_i\ x_{ik}$。$\varepsilon$是采样率，因为$0<r_k(z)<1$，所以我们会得到$1/\varepsilon$个分界点。

![6](https://raw.githubusercontent.com/Dounm/TheFarmOfDounm/master/resources/images/gdbt/6.png)
（该图未考虑权重$h$）

### 2.2 Sparsity-aware Split Finding

实际运用中，输入$x$很可能是稀疏的。有很多种可能造成数据是稀疏的：1) 数据中的缺失值； 2）较多的0值；3) 类似one-hot的特征工程。

对于这些缺失值，xgboost将样本分类到默认分支上去，而默认分支是由non-missing value学习得到的。具体算法如下：

![7](https://raw.githubusercontent.com/Dounm/TheFarmOfDounm/master/resources/images/gdbt/7.png)

由上可见，该算法**只考虑non-missing entries $I_k$**，因此计算复杂度是数据中的非缺失值个数的线性比值，对于稀疏数据来说，计算的很快。

对于缺失值的样本，算法分别将它们放到左节点和右节点，选取增益最大的一侧作为默认分类。

### 2.3 Column Block

XGBoost存储训练数据的数据结构被称为Block，每个Block内的数据采用**CSC格式（Compressed Sparse Column）**存储。

什么是CSC格式？CSC格式使用3个数组来存储稀疏矩阵。3个数组的定义如下：

- values[]：存储non-zero entry的值，按照column-major的顺序
- rowIndices[]：对于values[]中的每个元素，存储其对应的行的下标
- colPtrs[]：colPtrs[0]=0, colPtrs[i]=colPtrs[i-1]+(原始矩阵中第i-1列非0元素的个数)

举例如下：
$$
\begin{matrix}
9&0&1 \\
0&8&0 \\
0&6&0
\end{matrix}
$$
存储的3个矩阵分别是：
$$
values[]=9,8,6,1 \\
rowIndices[]=0,1,2,0 \\
colPtrs[]=0,1,3,4
$$
实际上xgboost存储的时候，每一列都按照该列特征的取值排序好。这样一来的话，在寻找最优分割点，遍历所有特征的取值时，我们只需要遍历$values[]​$即可（可以多线程并行加速遍历不同列的特征）。

### 2.4 分布式实现

多机实现中，每个worker节点都会启动一个XGBoost进行，进程之间互相通信的数据包括：树模型的最新参数；每次分裂叶子节点时，为了计算最优split point，所需要从各个节点汇集的统计量，包括近似算法中的bucket信息等。各节点先将自身计算得到的信息传给Rank0节点，Rank0节点汇总后把树模型最新参数发给各个节点。


## Reference
1. 《统计学习方法》 李航
2. [机器学习-一文理解GBDT的原理-20171001](https://zhuanlan.zhihu.com/p/29765582)
3. [GBDT算法原理深入解析](https://www.zybuluo.com/yxd/note/611571)
4. [Gradient Boosting Decision Tree](http://willzhang4a58.github.io/2016/06/gbdt/)
5. [机器学习算法中GBDT和XGBOOST的区别有哪些？](https://www.zhihu.com/question/41354392)
6. Chen, Tianqi, and Carlos Guestrin. "Xgboost: A scalable tree boosting system." Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining. ACM, 2016.
7. [XGboost: A Scalable Tree Boosting System论文及源码导读](http://mlnote.com/2016/10/05/a-guide-to-xgboost-A-Scalable-Tree-Boosting-System/)
8. [Why xgboost is so fast？-Yafei Zhang](https://pan.baidu.com/s/1hrVWQrU?errno=0&errmsg=Auth%20Login%20Sucess&&bduss=&ssnerror=0&traceid=)