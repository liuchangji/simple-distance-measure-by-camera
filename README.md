# simple-distance-measure-by-camera
基于地面假设的单目测距，核心代码文件

由于yolov5的代码在不断更新，我在这里放出核心部分的代码，原理十分简单，我告诉大家怎么用

视频：https://www.bilibili.com/video/BV1oh411a71G

首先我是使用yolov5完成这个功能的，你要先配置好yolov5
# 计算距离.py
这个文件中是计算距离的函数，设置好相机内参，高度，还有角度即可，把中文名改了，改个英文的，方便你调用这个文件

这里的函数会同时绘制出结果
# depth_detect_in_cam.py
这个文件是从yolov5的detect.py改的，就是把上面计算距离的函数import进来，我是在131行调用的。你需要根据最新的yolov5代码，放置并调用测距函数
 
 
 yolov5版本经常更新，我不建议你把这个文件直接放到yolov5的文件夹里使用，而是自己修改detect.py,添加计算距离.py中的测距函数

# 最后

至此，结束，写的很垃圾，但是很多人想用，模型本身也超级简单，就放出核心代码，学习一下即可，以后会好好完善，而且最近也在研究如何使用图像来估计地面的参数，来自动修正地面角度

# 一个坑
openCV调用摄像头会自动改变他的分辨率，如果跟你标定时的分辨率不一样，会出现错误哦

使用这个来设置opencv的分辨率


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)


cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

# 参考资料
相关文章《Vision-based ACC with a Single Camera: Bounds on Range and Range Rate Accuracy》
参考资料：单镜头视觉系统检测车辆的测距方法 - 知乎 https://zhuanlan.zhihu.com/p/57004561
