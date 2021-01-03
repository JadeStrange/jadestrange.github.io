---
title: K-means Python3实现
description: K-means Python3实现,adaboost,谱聚类思想,K-means混合高斯解释
categories:
 - 作业
tags:
 - 聚类

---

<!--> more注释 -->

### 1. 请简述 adaboost 算法的设计思想和主要计算步骤。

设计思想：

1.给定训练集，寻找比较粗糙的分类规则（弱分类器），要比寻找精确的分类规则要简单得多。

2.提升算法的核心，从弱学习算法出发，反复学习，得到一系列弱分类器，然后组合这些弱分类器，构成一个强分类器。

3.基本做法：改变训练数据的概率(权重)分布，针对不同的训练数据的分布，调用弱学习算法来学习一系列分类器。

其中有两个关键实现：

1.在每轮训练中，如何改变训练数据的权值或者分布？Adaboost提高那些前一轮弱分类器分错的样本的权重，降低已经被正确分类的样本的权重。错分的样本将在下一轮弱分类器中得到更多的关注。于是分类问题被一系列弱分类器分而治之。

2.如何将一系列的弱分类器组合成一个强分类器？采用加权(多数)表决的方法。具体地，加大分类错误率较小的弱分类器的权重，使其在表决中起到更大作用。

主要计算步骤：

输入训练数据集：$T=\{(x_1,y_1), (x_2,y_2), .., (x_n, y_n) \}$

输入弱学习算法：

​	（1）初始化数据训练权重分布$D_1=\{w_{11},w_{12}, .. w_{1n}\}$

​	（2）分别对m=1,2,...,M进行操作，练M个弱训练器

​			（2a）$G_m(x):->\{-1, +1\}$用权重分布$D_m$学习基本分类器

​			（2b）计算$G_m(x)$上的分类错误率，加权$e_m=P(G_m(x)\neq y_i)=\sum_{i=1}^{n}w_{mi}I(G_m(x_i)\neq y_i)$

​			（2c）计算$G_m(x)$贡献函数$\alpha_m=\frac{1}{2}ln\frac{1-e_m}{e_m}$

​			（2d）更新$D_{m+1}$

​	（3）构造基本线性组合$f(x)=\sum_{m=1}^M\alpha_mG_m(x)$，之后得到$G(x)=sign(f(x))$

### 2. 请从混合高斯密度函数估计的角度，简述K-Means聚类算法的原理(请主要用文字描述， 条理清晰)；请给出 K-Means 聚类算法的计算步骤；请说明哪些因素会影响 K-Means 算 法的聚类性能。

对于混合高斯密度函数估计引入如下假设：

1.各类出现的先验概率均相等。

2.每个样本点以概率为1属于一个类(后验概率0-1近似)；

​计算数据点到类中心的欧氏距离的平方，即计算$\|\|x_k-\hat\mu_i\|\|$，寻找与样本$x_k$最近的类中心点，将$x_k$分给最近的类:

​当$x_k距离\hat\mu_i最近时$$\hat P(\omega_i\|x_k,\hat\theta) \approx1$，否则为0

基于上述假设，对于c个高斯分布的均值，我们有：
$$
\hat \mu_i=\frac{\sum_{k=1}^{n}P(\omega_i|x_k,\hat\mu)x_k}{\sum_{k=1}^nP(\omega_i|x_k,\hat\mu)} = \frac{1}{n}\sum_{x_k \in \omega_i}x_k \ \ \ i=1,2,...,c
$$


但是，样本$x_k$最终得到$c$个高斯分布的均值之后，以这些均值作为c个类中心，计算每个样本点到类中心的欧氏距离，将样本点归入到距离最近的类，从而完成K_均值聚类工作。

影响Kmeans算法性能的有聚类簇数量多少，初始点位置，数据中的噪声点、孤立点数量，数据簇形状等。

### 3. 请简述谱聚类算法的原理，给出一种谱聚类算法（经典算法、Shi 算法和 Ng 算法之一） 的计算步骤；请指出哪些因素会影响聚类的性能。

原理：

从图切割的角度，聚类就是要找到一种合理的分割图的方法，分割后能形成若干个子图。链接不同子图的边的权重尽可能小，子图内部边的权重尽可能大。

谱聚类算法建立在图论中的谱图理论基础上，其本质是将聚类问题转化为一个图上的关于顶点划分的最优问题。建立在点对亲和性基础上，理论上能对任意分布形状的样本空间进行聚类。

构造拉普拉斯矩阵L，图的连通子图与L矩阵特征值的关系如下，设G为一个具有非负连接权重的无向图，由图G导出的L的零特征值的重数等于G的连通子图的个数，但实际上，数据簇之间可能相互混杂重叠，所以L通常不具有分块形状（无论怎么调整顺序）因此，可以考察其中较小的几个特征值(k个)对应的特征向量。

核心过程：

- 利用点对之间的相似性，构造亲和度矩阵
- 构建拉普拉斯矩阵
- 求解拉普拉斯矩阵最小特征值对应的特征向量(通常会舍弃零特征所对应的分量全相等的特征向量)
- 有这些特征向量构成样本的新的特征，采用Kmeans等聚类方法完成最后的聚类。

Ng算法：

1. 输入，相似矩阵W，k个聚类簇
2. 计算$L_{sym}=D^{-1/2}LD^{-1/2}$_
3. 计算$L_{sym}$特征值最小的k个特征向量
4. 让$U\in R^{n\times k}$包含所有这个k个特征向量。
5. 从$U$构造矩阵$T\in R^{n\times k}$，normalizing每行到二范数为1，让$t_{ij}=\frac{u_{ij}}{\sqrt{\sum_{m=1}^nu_{im}^2}}$
6. 让每行作为一个样本
7. 对其进行Kmeans聚类得到$A_1,A_2,...,A_k$
8. 输出$A_1,A_2,...,A_k$

影响聚类性能的因素：

1. 选择的k的数量，即选择特征向量的数量。
2. 亲和度矩阵的构建方式
3. 最后进行聚类所选择的方法。

## 第二部分：计算机编程

### 1．现有 1000 个二维空间的数据点，可以采用如下 MATLAB 代码来生成：

在运行完上述代码之后，可以获得 1000 个数据点，它们存储于矩阵X之中。X 是一个行数为 1000列数为2的矩阵。即是说，矩阵X 的每一行为一个数据点。另外，从上述 MATLAB 中可见，各真实分布的均值向量分别为 mu1, mu2, mu3, mu4, mu5。 提示：在实验中，生成一个数据矩阵X之后，就将其固定。后续实验均用此数据集，以便于分析算法。

### (1). 编写一个程序，实现经典的 K-均值聚类算法；

```python
def kmeans(cluster_n, X):
    """
    kmeans
    :param cluster_n: 分类数量
    :param X: 特征
    :return:
    """
    # 加载数据
    temp_max = np.max(X, axis=0)
    temp_min = np.min(X, axis=0)
    # 初始化
    cluster_center_point_list = []
    for i in range(cluster_n):
        x = random.uniform(temp_min[0], temp_max[0])
        y = random.uniform(temp_min[1], temp_max[1])
        cluster_center_point_list.append(np.array([x, y]))
    # 循环
    label_array = np.ones(X.shape[0], dtype=np.int) * (-1)

    epoch = 0
    while True:

        epoch += 1
        for i in range(X.shape[0]):
            norm2_list = np.array([np.linalg.norm(cluster_center_point_list[cluster_i] - X[i], 2)
                                   for cluster_i in range(cluster_n)])
            temp_cluster_id = norm2_list.argmin()
            label_array[i] = temp_cluster_id

        new_cluster_center_point_sum_list = [np.array([0.0, 0.0]) for _ in range(cluster_n)]
        new_cluster_center_point_num_list = [0 for _ in range(cluster_n)]
        for data_id, label_id in enumerate(label_array):
            new_cluster_center_point_sum_list[label_id] = new_cluster_center_point_sum_list[label_id] + X[
                data_id]
            new_cluster_center_point_num_list[label_id] += 1

        new_cluster_center_point_list = []
        for cluster_center_id, cluster_center_point in enumerate(new_cluster_center_point_sum_list):

            if new_cluster_center_point_num_list[cluster_center_id] != 0:
                cluster_center_point = cluster_center_point / new_cluster_center_point_num_list[cluster_center_id]
            new_cluster_center_point_list.append(cluster_center_point)

        cluster_point_equal = True
        for cluster_id, new_cluster_center_point in enumerate(new_cluster_center_point_list):
            if new_cluster_center_point[0] != cluster_center_point_list[cluster_id][0] and \
                    new_cluster_center_point[1] != cluster_center_point_list[cluster_id][1]:
                cluster_point_equal = False
                break
        if cluster_point_equal:
            break
        else:
            cluster_center_point_list = new_cluster_center_point_list
    print("epoch:", epoch)
    return label_array, cluster_center_point_list
```

### (2)令聚类个数等于 5，采用不同的初始值，报告聚类精度、以及最后获得的聚类中心， 并计算所获得的聚类中心与对应的真实分布的均值之间的误差。

较好的情况，随机初始值1：

迭代次数epoch: 5
准确率Homogeneity: 0.969
聚类中心 [array([1.01415941, 3.96384996]), array([ 9.05324946, -0.04146593]), array([ 5.4918652 , -4.40789589]), array([6.12513501, 4.51401558]), array([ 1.12994159, -1.06260689])]
聚类中心与真值均值之间的误差: 0.013

<img src="D:\Desktop\模式识别第五次作业\Kmeans_results_1.png" style="zoom:72%;" />

较差的一个聚类情况，随机初始值2：

epoch: 13
Homogeneity: 0.767
聚类中心 [array([ 5.32243428, -4.32696178]), array([9.06047487, 0.70167894]), array([5.80683471, 4.56049797]), array([ 9.00073177, -1.24803554]), array([0.94192585, 1.36741968])]
聚类中心与真值均值之间的误差: 1.252

<img src="D:\Desktop\模式识别第五次作业\Kmeans_results_2.png" alt="f" style="zoom:72%;" />