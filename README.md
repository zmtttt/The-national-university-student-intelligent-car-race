第十六届全国大学生智能车竞赛百度赛道代码开源
=============================================
本团队对地标及侧面任务共标注近4000张图片。使用labelimg标注，格式为voc。

标注的地标数据集已共享在aistudio：https://aistudio.baidu.com/aistudio/projectdetail/2170723 。

标注的侧面数据集已共享在aistudio：https://aistudio.baidu.com/aistudio/projectdetail/2170794 。

巡线数据集已共享在aistudio：https://aistudio.baidu.com/aistudio/datasetdetail/108767。

小车运行主程序在src内
------
使用control_collect_line_data.py或nocontrol_collect_line_data.py请放在src内运行

要想出色的完成所有任务的一个必要的前提就是小车能够完成任务一——自主续航，这是执行所有任务的前提。因此，我们团队花费大多时间和精力在任务一上面。我们采用的方案是用深度学习的方法进行训练来得到巡线模型，所以模型的好坏很大程度上是和数据集有关系的，因此我们的第一步车道线数据的采集工作十分重要。


对于车道线的采集传统的方法是利用手柄来遥控小车采集一系列的数据，在小车行驶的过程中我们用摄像机拍摄图片，采用归一化的方法来获得小车在行驶过程中偏离中心线的角度，将所有的角度值保存成一个json文件，将所有的图片以及角度值放到我们的神经网络当中进行训练，即可生成一个模型。模型生成好部署在小车之后让小车自主巡航的效果并不好，我们分析其中的原因发现小车对于光照的条件较为苛刻，尤其是在强光源照射时，我们发现地图会产生严重的反光现象，导致地图上的白斑范围较大，摄像头根本无法看清地图上的内容，因此就无法根据地图上的信息推理出小车偏离中心线的角度，导致小车巡航任务的失败。并且小车在高速度巡航时摄像机所拍摄的图片会出现模糊的现象，也有可能导致小车巡航任务的失败。而且在使用手柄采集数据的过程中，我们发现触动遥控器上的摇杆进行遥控时，小车所返回的角度都会较上一次有较大的波动，导致模型训练好之后小车会出现严重的摆尾现象，尤其是在经过环岛以及比较急的转弯时。

为了解决上述问题，团队成员将上述方法进行了优化。对于问题一，在硬件方面我们在摄像头上安装了偏振片，减弱光在地图上的反射。在车道线数据集中我们加入了不同光照强度下、不同车速下小车采集的图片，以此增强车在续航过程中的鲁棒性。对于问题二，我们的解决方案是不采用手柄的遥控来采集车道线数据集，而是利用颜色提取来找到赛道的中心线，让智能汽车按此中心线来进行巡航并采集数据集。采用这种方法的好处是在直线行驶时智能汽车上下帧图像所对应的角度值不会变化很大，并且在转弯时很流畅，在转弯之后车身也不会有太大的抖动。

![hsv颜色提取](https://user-images.githubusercontent.com/90136090/133626826-2b1f99d4-d0ce-465f-8fb0-2ef6cce038d1.jpg)

### hsv颜色提取

特别是问题二，在采集图片时为了减少光的影响，我们使用了在hsv空间提取颜色，并通过PD（没有I）控制来获得转弯角度，公式如下：

differ=(x×p+d×(x-last_x))/120

其中differ为转弯角度，x为当前整个图片的中点与两侧黄线的中点像素差，last_x为上一帧的整个图片的中点与两侧黄线的中点像素差，经过我们大量实践，p的取值为1.05，d的取值为0.098

因此在良好的巡线基础上我们取得了东部赛区第七名，最终获得全国一等奖。并且在随后的调试中，小车以70的速度（满速为100）能在半分钟内跑完整个赛道,并对任务优化后整个完成所有任务能在一分半内完成。
