import cv2
import numpy as np
import math
import pandas as pd


def cv_imread(path):
    cv_img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    return cv_img


def entire_image_matching_sift(name1, name2):
    # 读取图片
    img1 = cv2.imread(name1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(name2, cv2.IMREAD_GRAYSCALE)

    # 用ORB找特征点及其描述子
    # orb = cv2.ORB_create()
    # orb = cv2.ORB.create()
    sift = cv2.SIFT.create()

    key1, des1 = sift.detectAndCompute(img1, None)
    key2, des2 = sift.detectAndCompute(img2, None)

    # 特征点匹配
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(des1, des2, 2)

    ratio_thres = 0.6
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thres * n.distance:
            good_matches.append(m)

    # 匹配点对在照片上的位移
    match_point_pairs = []
    for i in range(0, len(good_matches)):
        left = key1[good_matches[i].queryIdx].pt
        right = key2[good_matches[i].trainIdx].pt
        match_point_pairs.append([left, right])

    dist = []
    for i in range(0, len(match_point_pairs)):
        dist.append(math.sqrt(pow(match_point_pairs[i][0][0] - match_point_pairs[i][1][0], 2) +
                              pow(match_point_pairs[i][0][1] - match_point_pairs[i][1][1], 2)))

    # 画出两幅图像及匹配点
    img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
    cv2.drawMatches(img1, key1, img2, key2, good_matches, img_matches,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imshow("matches", img_matches)
    cv2.waitKey()


def entire_image_matching_orb(name1, name2):
    # 读取图片
    img1 = cv2.imread(name1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(name2, cv2.IMREAD_GRAYSCALE)

    # 用ORB找特征点及其描述子
    orb = cv2.ORB_create()
    # orb = cv2.ORB.create()
    # sift = cv2.SIFT.create()

    key1, des1 = orb.detectAndCompute(img1, None)
    key2, des2 = orb.detectAndCompute(img2, None)

    # 创建KNN匹配器
    index_params = dict(algorithm = 6,
                        table_number = 6,
                        key_size = 12,
                        multi_probe_level = 1)
    search_params = dict()
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    # matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    # matcher = cv2.DescriptorMatcher.create(cv2.DescriptorMatcher_FLANNBASED)

    # 匹配
    matches = matcher.knnMatch(des1, des2, k=2)

    good_matches = []
    ratio_thres = 0.7
    for match in matches:
        # if m.distance < ratio_thres * n.distance:
        #     good_matches.append(m)
        if len(match) <= 1:
            continue
        else:
            if match[0].distance < ratio_thres * match[1].distance:
                good_matches.append(match[0])

    # 匹配点对在照片上的位移
    match_point_pairs = []
    for i in range(0, len(good_matches)):
        left = key1[good_matches[i].queryIdx].pt
        right = key2[good_matches[i].trainIdx].pt
        match_point_pairs.append([left, right])

    dist = []
    for i in range(0, len(match_point_pairs)):
        dist.append(math.sqrt(pow(match_point_pairs[i][0][0] - match_point_pairs[i][1][0], 2) +
                              pow(match_point_pairs[i][0][1] - match_point_pairs[i][1][1], 2)))

    # 画出两幅图像及匹配点
    img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
    cv2.drawMatches(img1, key1, img2, key2, good_matches, img_matches,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imshow("matches", img_matches)
    cv2.waitKey()


def image_matching_orb(img1, img2):
    """
    通过orb算法计算特征点及描述子
    :param img1:
    :param img2:
    :return: boolean 是否找到对应特征点
    """

    # 用ORB找特征点及其描述子
    orb = cv2.ORB_create()
    # orb = cv2.ORB.create()
    # sift = cv2.SIFT.create()

    key1, des1 = orb.detectAndCompute(img1, None)
    key2, des2 = orb.detectAndCompute(img2, None)

    # 若无特征点，返回错误
    if len(key1) == 0 or len(key2) == 0:
        return False

    # 创建KNN匹配器
    index_params = dict(algorithm=6,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=1)
    search_params = dict()
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    # matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    # matcher = cv2.DescriptorMatcher.create(cv2.DescriptorMatcher_FLANNBASED)

    # 匹配
    matches = matcher.knnMatch(des1, des2, k=2)

    good_matches = []
    ratio_thres = 0.6
    # for m, n in matches:
    for match in matches:
        # if m.distance < ratio_thres * n.distance:
        #     good_matches.append(m)
        if len(match) <= 1:
            continue
        else:
            if match[0].distance < ratio_thres * match[1].distance:
                good_matches.append(match[0])

    # 若通过测试的匹配点过少，返回错误
    if len(good_matches) < 10:
        return False

    # 匹配点对在照片上的位移
    match_point_pairs = []
    for i in range(0, len(good_matches)):
        left = key1[good_matches[i].queryIdx].pt
        right = key2[good_matches[i].trainIdx].pt
        match_point_pairs.append([left, right])

    dist = []
    x_diff = []
    y_diff = []
    for i in range(0, len(match_point_pairs)):
        this_x_diff = match_point_pairs[i][1][1] - match_point_pairs[i][0][1]
        this_y_diff = match_point_pairs[i][1][0] - match_point_pairs[i][0][0]
        x_diff.append(this_x_diff)
        y_diff.append(this_y_diff)
        dist.append(math.sqrt(pow(this_x_diff, 2) + pow(this_y_diff, 2)))

    x_diff = np.array(x_diff)
    y_diff = np.array(y_diff)
    dist = np.array(dist)

    # 正态分布拟合
    # goodness_of_fit(x_diff)
    # goodness_of_fit(y_diff)
    # goodness_of_fit(dist)
    _frequency_table(dist)

    # 画出两幅图像及匹配点
    img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
    cv2.drawMatches(img1, key1, img2, key2, good_matches, img_matches,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imshow("matches", img_matches)
    cv2.waitKey()

    return True


def image_matching_sift(img1, img2):
    """
    对一对影像进行SIFT匹配。
    若匹配的特征点小于20个，则返回空pandas字典
    完成匹配后，返回频数分布表
    :param img1:
    :param img2:
    :return:
    """

    # 用ORB找特征点及其描述子
    # orb = cv2.ORB_create()
    # orb = cv2.ORB.create()
    sift = cv2.SIFT.create()

    key1, des1 = sift.detectAndCompute(img1, None)
    key2, des2 = sift.detectAndCompute(img2, None)

    # 若无特征点，返回错误
    if len(key1) == 0 or len(key2) == 0:
        return pd.DataFrame(columns=['upperbound', 'actual frequency'])

    # 特征点匹配
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(des1, des2, 2)

    ratio_thres = 0.6
    good_matches = []

    for match in knn_matches:
        if len(match) <= 1:
            continue
        else:
            if match[0].distance < ratio_thres * match[1].distance:
                good_matches.append(match[0])

    if len(good_matches) < 20:
        return pd.DataFrame(columns=['upperbound', 'actual frequency'])

    # 匹配点对在照片上的位移
    match_point_pairs = []
    for i in range(0, len(good_matches)):
        left = key1[good_matches[i].queryIdx].pt
        right = key2[good_matches[i].trainIdx].pt
        match_point_pairs.append([left, right])

    dist = []
    for i in range(0, len(match_point_pairs)):
        dist.append(math.sqrt(pow(match_point_pairs[i][0][0] - match_point_pairs[i][1][0], 2) +
                              pow(match_point_pairs[i][0][1] - match_point_pairs[i][1][1], 2)))

    res = _frequency_table(dist)

    # 画出两幅图像及匹配点
    # img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
    # cv2.drawMatches(img1, key1, img2, key2, good_matches, img_matches,
    #                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #
    # cv2.imshow("matches", img_matches)
    # cv2.waitKey()

    return res


def split_image_and_orb(img1, img2):

    # 图像改为灰度值
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 图片大小
    img1_row = img1.shape[0]
    img1_col = img1.shape[1]

    img2_row = img2.shape[0]
    img2_col = img2.shape[1]

    # 分割图片
    for i in range(0, 4):

        # 左上角
        if i == 0:
            print("左上1/16")
            split_img1 = img1[0:int(img1_row/4), 0:int(img1_col/4)]
            split_img2 = img2[0:int(img2_row/4), 0:int(img2_col/4)]

        # 右上角
        elif i == 1:
            print("右上1/16")
            split_img1 = img1[0:int(img1_row/4), int(img1_col * 3/4):]
            split_img2 = img2[0:int(img2_row/4), int(img2_col * 3/4):]

        # 左下角
        elif i == 2:
            print("左下1/16")
            split_img1 = img1[int(img1_row * 3/4):, 0:int(img1_col/4)]
            split_img2 = img2[int(img2_row * 3/4):, 0:int(img2_col/4)]

        # 右下角
        elif i == 3:
            print("右下1/16")
            split_img1 = img1[int(img1_row * 3/4):, int(img1_col * 3/4):]
            split_img2 = img2[int(img2_row * 3/4):, int(img2_col * 3/4):]

        # ORB特征提取及匹配
        flag = image_matching_orb(split_img1, split_img2)
        print(flag)

        # 若边角没有特征点
        if not flag:

            # 左上角
            if i == 0:
                print("左上1/4")
                split_img1 = img1[0:int(img1_row / 2), 0:int(img1_col / 2)]
                split_img2 = img2[0:int(img2_row / 2), 0:int(img2_col / 2)]

            # 右上角
            elif i == 1:
                print("右上1/4")
                split_img1 = img1[0:int(img1_row / 2), int(img1_col / 2):]
                split_img2 = img2[0:int(img2_row / 2), int(img2_col / 2):]

            # 左下角
            elif i == 2:
                print("左下1/4")
                split_img1 = img1[int(img1_row / 2):, 0:int(img1_col / 2)]
                split_img2 = img2[int(img2_row / 2):, 0:int(img2_col / 2)]

            # 右下角
            elif i == 3:
                print("右下1/4")
                split_img1 = img1[int(img1_row / 2):, int(img1_col / 2):]
                split_img2 = img2[int(img2_row / 2):, int(img2_col / 2):]

            flag = image_matching_orb(split_img1, split_img2)
            print(flag)


def split_image_and_sift(img1, img2):
    """
    分割影像为16部分，对四角进行sift算法匹配，若特征点过少或其中一张图没有匹配点，则重新对图进行四等分再进行SIFT算法匹配
    四角全部完成后将四个频数分布表输出
    img1为后时相影像，img2为前时相裁剪影像
    """

    # 图像改为灰度值
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 图片大小
    img1_row = img1.shape[0]
    img1_col = img1.shape[1]

    img2_row = img2.shape[0]
    img2_col = img2.shape[1]


    # 左上角
    print("左上1/16")
    split_img1 = img1[0:int(img1_row / 4), 0:int(img1_col / 4)]
    split_img2 = img2[0:int(img2_row / 4), 0:int(img2_col / 4)]
    top_left = image_matching_sift(split_img1, split_img2)

    if top_left.shape[0] == 0:
        print("左上1/4")
        split_img1 = img1[0:int(img1_row / 2), 0:int(img1_col / 2)]
        split_img2 = img2[0:int(img2_row / 2), 0:int(img2_col / 2)]
        top_left = image_matching_sift(split_img1, split_img2)


    # 右上角
    print("右上1/16")
    split_img1 = img1[0:int(img1_row / 4), int(img1_col * 3 / 4):]
    split_img2 = img2[0:int(img2_row / 4), int(img2_col * 3 / 4):]
    top_right = image_matching_sift(split_img1, split_img2)

    if top_left.shape[0] == 0:
        print("右上1/4")
        split_img1 = img1[0:int(img1_row / 2), int(img1_col / 2):]
        split_img2 = img2[0:int(img2_row / 2), int(img2_col / 2):]
        top_right = image_matching_sift(split_img1, split_img2)


    # 左下角
    print("左下1/16")
    split_img1 = img1[int(img1_row * 3 / 4):, 0:int(img1_col / 4)]
    split_img2 = img2[int(img2_row * 3 / 4):, 0:int(img2_col / 4)]
    bottom_left = image_matching_sift(split_img1, split_img2)

    if bottom_left.shape[0] == 0:
        print("左下1/4")
        split_img1 = img1[int(img1_row / 2):, 0:int(img1_col / 2)]
        split_img2 = img2[int(img2_row / 2):, 0:int(img2_col / 2)]
        bottom_left = image_matching_sift(split_img1, split_img2)


    # 右下角
    print("右下1/16")
    split_img1 = img1[int(img1_row * 3 / 4):, int(img1_col * 3 / 4):]
    split_img2 = img2[int(img2_row * 3 / 4):, int(img2_col * 3 / 4):]
    bottom_right = image_matching_sift(split_img1, split_img2)

    if bottom_right.shape[0] == 0:
        print("右下1/4")
        split_img1 = img1[int(img1_row / 2):, int(img1_col / 2):]
        split_img2 = img2[int(img2_row / 2):, int(img2_col / 2):]
        bottom_right = image_matching_sift(split_img1, split_img2)

    return [top_left, top_right, bottom_left, bottom_right]


def _frequency_table(dist):
    """
    匹配点像素距离频率表
    :param: dist: 匹配点像素距离
    """
    std_x = np.std(dist)
    if std_x == 0:
        print("all 0")
        return

    # 计算区间
    interval_width = std_x                # 区间宽度
    min_res = np.min(dist)                      # 最小距离
    max_res = np.max(dist)                      # 最大距离

    low_bound = math.ceil(min_res / interval_width)
    up_bound = math.ceil(max_res / interval_width)

    interval = []
    for i in range(low_bound, up_bound+1):
        interval.append(i * interval_width)

    # 计算距离频率
    actual_freq = []
    for i in range(0, len(interval)):
        actual_freq.append((dist < interval[i]).sum() - (dist < (interval[i]-interval_width)).sum())

    # 频率表
    result_tb = pd.DataFrame({'upperbound': interval, 'actual frequency': actual_freq})
    print(result_tb)
    return result_tb


if __name__ == '__main__':
    # ImageMatching("MLC.img", "final_output.img")
    # ImageMatchingSIFT("2019image.jpeg", "2020image.jpeg")
    # ImageMatchingSIFT("feidong_image.jfif", "feidong_noise_image.jpeg")
    # ImageMatchingORB("feidong_image.jfif", "feidong_noise_image.jpeg")
    split_imgage_and_orb("feidong_image.jfif", "feidong_noise_image.jpeg")