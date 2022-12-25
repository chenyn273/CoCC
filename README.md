The replication package of paper _Are your comments outdated? Towards automatically detecting code-comment consistency_.
## 下载资源包
谷歌云盘链接：

*资源说明*
features：包含已经处理过的method和block类型的CCSet数据集，您可以在这里查看我们用到的数据集。

word2vector：包含词向量训练语料库及训练好的词向量模型

OCD.zip：训练好的OCD模型

ocd_data.zip：用来做baseline OCD的数据集，与我们项目用的数据集是同一个，仅输入形式不同。

RQ4：RQ4用到的数据集，包含ejbca、freecol、opennms三个项目，其中detected_outdated.txt是检测出的过时注释文件列表，他们可以分别在*.feature文件夹中找到。\*.git文件夹保存了项目原始的git clone文件
## 训练并测试模型
运行outdated_predict文件夹下main.py
这里包含网格搜索等，耗时较久
## 分类器比较及校准
运行classifiers文件夹下main.py
## utils
1.convert_co_CCSet，用来从原始commit生成CCSet的工具

2.get_change，用来从CCSet提取变更的工具

3.extract_features，用来提取特征的工具

4.generate_table，用来生成csv文件的工具

5.data_analyse_tool，用来分析数据集的工具

6.drop_duplicates_tool，用来对原始数据去重的工具

7.word2vector，用来训练词向量的工具
## Baseline
*基于规则的baseline：*
将资源包中的features放到CoCC代码目录下，然后运行baseline/rule/main.py
*OCD*
打开CoCC/baseline/OCD文件夹，将OCD.zip和ocd_data.zip解压到当前目录，然后命令行运行

python -m infer --log-dir OCD --config configs/OCD.yml

python -m eval --log-dir OCD --config configs/OCD.yml
