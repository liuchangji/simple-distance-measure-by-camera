# simple-distance-measure-by-camera
基于地面假设的单目测距，核心代码文件
由于yolov5的代码在不断更新，我在这里放出核心部分的代码，原理十分简单，我告诉大家怎么用
# 计算距离.py
这个文件中是计算距离的函数，设置好相机内参，高度，还有角度即可
# depth_detect_in_cam.py
这个文件是从yolov5的detect.py改的，就是把上面计算距离的函数import进来，我是在131行调用的。

至此，结束，写的很垃圾，但是很多人想用，模型本身也超级简单，就放出核心代码，学习一下即可
# 最后
最后只需要把depth_detect_in_cam.py放到yolov5的根目录里就OK了
