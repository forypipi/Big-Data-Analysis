# Big-Data-Analysis
大数据分析技术与实践课程相关作业代码、CHAOS数据集和医疗小样本分割实践代码

数据来自于grand-challenge Combined Healthy Abominal Organ Segmentation（CHAOS）竞赛，链接：https://chaos.grand-challenge.org/Combined_Healthy_Abdominal_Organ_Segmentation/

数据目录为C:\Users\orfu\Desktop\MyData\BaiduSyncdisk\研究方向\小样本学习\数据集\CHAOS

**2023.11.8 16:00更新**

上传两个jupyter notebook文件：

DescribeAnalysis.ipynb主要是对三维CT图像进行直方图分析，肝脏部分分割区域的直方图可以显著与其他区域分开，同时我们也可以看到有直方图0~200区间有多个峰，可能是不同器官的直方图区域不同。

ExploratoryAnalysis.ipynb主要包含一个交互可视化模块（在sitk官方代码基础上修改，官方代码不知道为什么点交互无法更新图片，疑似是没有清除画布上的内容，修改为了删除整个fig并重新创建，这样做的缺点主要是每次交互都会新出来一张图，不会删除之前的图），除此以外还提取了分割区域的一阶二阶等影像组学特征，用于语义分割的01mask相关性分析，但是遇到了非mask区域计算速度很慢，有的会报错的问题。

**2023.11.8 20:00更新**

上传了本次作业全部代码文件，包括交互可视化界面、CT分布直方图、特征-标签箱线&小提琴图（CorrelationAnalysis.ipynb）。
