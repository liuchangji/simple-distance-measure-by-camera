def draw_measure_line(xyxy, img, size, label, intrinsics_matrix, color=None):
    """
    :param xyxy:左上右下
    :param img:图像
    :param size 字体大小
    :param color:
    :param label:
    :param intrinsics_matrix:内参，intrinsics_matrix = [960, 540,775.9, 776.9]  # cx,cy,fx,fy
    :return:目标的位置
    """

    h = 1.45  # 相机地面高度 1.65 m
    alpha = 0 # 角度a

    # 这个这是绘制测距线用的，没什么用。你需要按照真实位置计算出地面线在图像中的位置，否则绘制出来也仅仅是个参考
    # 这个与你测量结果无关，不理解的话注释掉即可
    # limit_view_width = 10.9
    # limit_depth = 6.21


    u0 = intrinsics_matrix[0]  # cx
    v0 = intrinsics_matrix[1]  # cy

    fx = intrinsics_matrix[2]
    fy = intrinsics_matrix[3]

    pi = math.pi

    filter_list = [9, 11, 10, 74] # 过滤一些我不想要的类别
    if label not in filter_list:

        y = int(xyxy[3])
        x = (xyxy[0] + xyxy[2]) // 2
        color_ground_point = (255 - np.array(color)).tolist()
        cv2.circle(img, (x, y), 8, color_ground_point, thickness=-1)

        Q_pie = [x - u0, y - v0]
        gamma_pie = math.atan(Q_pie[1] / fy) * 180 / 3.14

        beta_pie = alpha + gamma_pie

        if beta_pie == 0:
            beta_pie = 0.01

        O1Q = round(h / math.tan(beta_pie / 180 * pi), 1)

        z_in_cam = (h / math.sin(beta_pie / 180 * pi)) * math.cos(gamma_pie * pi / 180)
        x_in_cam = z_in_cam * (x - u0) / fx
        x_in_cam = x_in_cam.cpu().numpy()
        y_in_cam = z_in_cam * (y - v0) / fy
        distance = round(math.sqrt(O1Q ** 2 + x_in_cam ** 2), 2)

        # 绘制测距线
        # temp = limit_depth * x_in_cam / z_in_cam
        # temp = temp / limit_view_width
        # connect_point_x = int((1 + temp) * img.shape[1] // 2)
        # line_color = color_map_fun(distance, 8, 3)
        # cv2.line(img, (x, y), (connect_point_x, img.shape[0]), line_color, thickness=2)

        if distance < 0:
            distance = "unknown"
        cv2.putText(img, str(distance) + 'm', (x + size, y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=size, color=line_color,
                    thickness=3)
        return np.asarray([x_in_cam, y_in_cam, z_in_cam])
