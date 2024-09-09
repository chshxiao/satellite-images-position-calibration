import arcpy

DIST_THRESHOLD = 400
PIXEL_COUNT_THRESHOLD = 10


def detect_shift_each_corner(freq_tb, cell_siz):
    """
    检测是否匹配点有否偏移超过阈值。
    若是，则返回True
    否则返回False
    :param freq_tb:
    :param cell_siz:
    :return:
    """

    # 根据定义的偏移阈值计算偏移像素阈值
    pixel_threshold = DIST_THRESHOLD / cell_siz


    # 找出所有多于10个匹配点的偏移区间，判断它们是否大于阈值
    selected_freq_tb = freq_tb[freq_tb['actual frequency'] > 10]
    for i in range(0, selected_freq_tb.shape[0]):
        if selected_freq_tb.iloc[i].at["upperbound"] > pixel_threshold:
            return True

    return False


def detect_shift(freq_tbs, cell_siz):
    """
    检测全图像匹配点有否偏移超过阈值。
    若是，则返回“影像正常”
    否则返回出现超过阈值偏移的部分，比如“图像左上偏移”
    :param freq_tbs:
    :param cell_siz:
    :return:
    """

    wrong_part = ""
    place_str = ("左上", "右上", "左下", "右下")

    for i in range(0, 4):
        flag = detect_shift_each_corner(freq_tbs[i], cell_siz)
        if flag:
            wrong_part = wrong_part + place_str[i]

    if wrong_part == "":
        return "影像正常"
    else:
        return "影像" + wrong_part + "偏移"