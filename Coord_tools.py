import cv2
import numpy as np
import math


def SI_calculate(img_path, landmarks):
    """
    Calculating Symmetry-Index using extracted facial landmarks (68 points)
    """
    # Reference points
    # pts = landmarks[1]
    save_path = img_path.split('.')[0]
    pts = landmarks
    ME_L, ME_R = pts[19], pts[24]  # Middle of Eyebrows: ME
    ET_L, ET_R = pts[37], pts[44]  # Eye Top: ET
    EB_L, EB_R = pts[41], pts[46]  # Eye Bottom: EB
    EOC_L, EOC_R = pts[36], pts[45]  # Eye Outer Corner: EOC
    EIC_L, EIC_R = pts[39], pts[42]  # Eye Inner Corner: EIC
    MC_L, MC_R = pts[48], pts[54]  # Mouth Corner: MC
    TN, RN1, RN2, Chin = pts[30], pts[28], pts[29], pts[8]  # Tip of the Nose: TN, Nose Root: RN, Chin
    RN_M = cross_point(np.append(RN1, RN2), np.append(MC_L, MC_R))  # Nose root to Mouth
    M_M_up = cross_point(np.append(TN, Chin), np.append(pts[50], pts[52]))  # Mouth upper middle point
    M_M_low = cross_point(np.append(TN, Chin), np.append(pts[58], pts[56]))  # Mouth lower middle point

    E_L_contour = [pts[36], pts[37], pts[38], pts[39], pts[40], pts[41]]  # Left eye contour
    E_L_contour1 = [pts[37], pts[38], pts[39], pts[40], pts[41], pts[36]]  # for line drawing
    E_R_contour = [pts[42], pts[43], pts[44], pts[45], pts[46], pts[47]]  # Right eye contour
    E_R_contour1 = [pts[43], pts[44], pts[45], pts[46], pts[47], pts[42]]  # for line drawing
    M_L_contour = [pts[48], pts[49], M_M_up, M_M_low, pts[58], pts[59]]  # Left mouth contour
    M_L_contour1 = [pts[49], M_M_up, M_M_low, pts[58], pts[59], pts[48]]  # for line drawing
    M_R_contour = [pts[54], pts[53], M_M_up, M_M_low, pts[56], pts[55]]  # Right mouth contour
    M_R_contour1 = [pts[53], M_M_up, M_M_low, pts[56], pts[55], pts[54]]  # for line drawing

    # Save images with lines
    img_ME_TN = draw_lines(img_path, [ME_L, ME_R], [TN, TN])
    img_ET_EB = draw_lines(img_path, [ET_L, ET_R], [EB_L, EB_R])
    img_EOC_EIC = draw_lines(img_path, [EOC_L, EOC_R], [EIC_L, EIC_R])
    img_MC_TN = draw_lines(img_path, [MC_L, MC_R], [TN, TN])
    img_Area_Eye = draw_lines(img_path, E_L_contour + E_R_contour, E_L_contour1 + E_R_contour1)
    img_Area_Mouth = draw_lines(img_path, M_L_contour + M_R_contour, M_L_contour1 + M_R_contour1)
    img_Angle_mouth = draw_lines(img_path, [RN1, MC_L], [RN_M, RN_M])
    cv2.imwrite(save_path + '_Eyebrows.jpg', img_ME_TN)
    cv2.imwrite(save_path + '_EyeHeight.jpg', img_ET_EB)
    cv2.imwrite(save_path + '_EyeWidth.jpg', img_EOC_EIC)
    cv2.imwrite(save_path + '_MouthCorner.jpg', img_MC_TN)
    cv2.imwrite(save_path + '_EyeArea.jpg', img_Area_Eye)
    cv2.imwrite(save_path + '_MouthArea.jpg', img_Area_Mouth)
    cv2.imwrite(save_path + '_MouthAngle.jpg', img_Angle_mouth)

    # Distance, Area, Angle calculation
    ME_TN_L, ME_TN_R = pts_distance(ME_L, TN), pts_distance(ME_R, TN)  # ME to TN (brows height)
    ET_EB_L, ET_EB_R = pts_distance(ET_L, EB_L), pts_distance(ET_R, EB_R)  # ET to EB (eyes height)
    EOC_EIC_L, EOC_EIC_R = pts_distance(EOC_L, EIC_L), pts_distance(EOC_R, EIC_R)  # EOC to EIC (eyes widths)
    Area_Eye_L, Area_Eye_R = GetAreaOfPolyGon(E_L_contour), GetAreaOfPolyGon(E_R_contour)
    Area_Mouth_L, Area_Mouth_R = GetAreaOfPolyGon(M_L_contour), GetAreaOfPolyGon(M_R_contour)
    MC_TN_L, MC_TN_R, Angle_Mouth = pts_distance(MC_L, TN), pts_distance(MC_R, TN), vec_angle(MC_L, MC_R, RN1, RN2)
    if Angle_Mouth == 90:
        M_L_contour = [pts[48], pts[49], pts[50], pts[51], pts[57], pts[58], pts[59]]  # Left mouth contour
        M_L_contour1 = [pts[49], pts[50], pts[51], pts[57], pts[58], pts[59], pts[48]]
        M_R_contour = [pts[54], pts[53], pts[52], pts[51], pts[57], pts[56], pts[55]]  # Right mouth contour
        M_R_contour1 = [pts[53], pts[52], pts[51], pts[57], pts[56], pts[55], pts[54]]
        Area_Mouth_L, Area_Mouth_R = GetAreaOfPolyGon(M_L_contour), GetAreaOfPolyGon(M_R_contour)
        img_Area_Mouth = draw_lines(img_path, M_L_contour + M_R_contour, M_L_contour1 + M_R_contour1)
        cv2.imwrite(save_path + '_MouthArea.jpg', img_Area_Mouth)

    # Symmetry-Index calculation
    D_Eyebrows, D_Eyes_height, D_Eyes_width, D_Eye_area, D_Mouth, D_Mouth_area = \
        abs(ME_TN_L - ME_TN_R), abs(ET_EB_L - ET_EB_R), abs(EOC_EIC_L - EOC_EIC_R), \
        abs(Area_Eye_L - Area_Eye_R), abs(MC_TN_L - MC_TN_R), abs(Area_Mouth_L - Area_Mouth_R)
    D_Eyebrows_ratio = ratio_cal(ME_TN_L, ME_TN_R)
    D_Eyes_height_ratio = ratio_cal(ET_EB_L, ET_EB_R)
    D_Eyes_width_ratio = ratio_cal(EOC_EIC_L, EOC_EIC_R)
    D_Eye_area_ratio = ratio_cal(Area_Eye_L, Area_Eye_R)
    D_Mouth_ratio = ratio_cal(MC_TN_L, MC_TN_R)
    D_Mouth_area_ratio = ratio_cal(Area_Mouth_L, Area_Mouth_R)
    D_Mouth_angle_ratio = ratio_cal(90, Angle_Mouth)
    D_Eyebrows_adj = gamma(D_Eyebrows_ratio, 1000, 2.5)
    D_Mouth_adj = gamma(D_Mouth_ratio, 300, 2.5)
    D_Mouth_area_adj = gamma(D_Mouth_area_ratio, 200, 2)
    D_Mouth_angle_adj = gamma(D_Mouth_angle_ratio, 1200, 2)
    D_Eyes_height_adj = gamma(D_Eyes_height_ratio, 200, 2.5)
    D_Eyes_width_adj = gamma(D_Eyes_width_ratio, 1500, 2.5)
    D_Eye_area_adj = gamma(D_Eye_area_ratio, 100, 2)
    SI_Eyebrows = 100 - D_Eyebrows_adj
    SI_Eyes = 100 - D_Eyes_height_adj - D_Eyes_width_adj - D_Eye_area_adj
    SI_Mouth = 100 - D_Mouth_adj - D_Mouth_angle_adj - D_Mouth_area_adj

    SI_data = [round(ME_TN_L, 2), round(ME_TN_R, 2), round(D_Eyebrows, 2), round(ET_EB_L, 2), round(ET_EB_R, 2),
               round(D_Eyes_height, 2), round(EOC_EIC_L, 2), round(EOC_EIC_R, 2), round(D_Eyes_width, 2),
               round(Area_Eye_L, 2), round(Area_Eye_R, 2), round(D_Eye_area, 2),
               round(MC_TN_L, 2), round(MC_TN_R, 2), round(D_Mouth, 2), round(Area_Mouth_L, 2), round(Area_Mouth_R, 2),
               round(D_Mouth_area, 2), round(Angle_Mouth, 2), round(SI_Eyebrows, 2), round(SI_Eyes, 2),
               round(SI_Mouth, 2)
               ]
    return SI_data


def draw_lines(img_path, data1, data2):
    img = cv2.imread(img_path)
    for i in range(len(data1)):
        img = cv2.line(img, tuple(map(int, data1[i])), tuple(map(int, data2[i])), (0, 255, 0), thickness=3)
    return img


def cross_point(line1, line2):
    x1 = line1[0]
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]

    x3 = line2[0]
    y3 = line2[1]
    x4 = line2[2]
    y4 = line2[3]

    k1 = (y2 - y1) * 1.0 / (x2 - x1)
    b1 = y1 * 1.0 - x1 * k1 * 1.0
    if (x4 - x3) == 0:
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)
        b2 = y3 * 1.0 - x3 * k2 * 1.0
    if k2 is None:
        x = x3
    else:
        x = (b2 - b1) * 1.0 / (k1 - k2)
    y = k1 * x * 1.0 + b1 * 1.0
    return [x, y]


def ratio_cal(data1, data2):
    min_num, max_num = min(data1, data2), max(data1, data2)
    return 1 - (min_num / max_num)


def GetAreaOfPolyGon(points):
    area = 0
    if len(points) < 3:
        raise Exception("error")

    p1 = points[0]
    for i in range(1, len(points) - 1):
        p2 = points[i]
        p3 = points[i + 1]

        # 计算向量
        # vecp1p2 = Point(p2.x - p1.x, p2.y - p1.y)
        vecp1p2 = [p2[0] - p1[0], p2[1] - p1[1]]
        # vecp2p3 = Point(p3.x - p2.x, p3.y - p2.y)
        vecp2p3 = [p3[0] - p2[0], p3[1] - p2[1]]

        vecMult = vecp1p2[0] * vecp2p3[1] - vecp1p2[1] * vecp2p3[0]  # 判断正负方向比较有意思
        sign = 0
        if vecMult > 0:
            sign = 1
        elif vecMult < 0:
            sign = -1

        triArea = GetAreaOfTriangle(p1, p2, p3) * sign
        area += triArea
    return abs(area)


def GetAreaOfTriangle(p1, p2, p3):
    p1p2 = GetLineLength(p1, p2)
    p2p3 = GetLineLength(p2, p3)
    p3p1 = GetLineLength(p3, p1)
    s = (p1p2 + p2p3 + p3p1) / 2
    area = s * (s - p1p2) * (s - p2p3) * (s - p3p1)  # 海伦公式
    area = math.sqrt(area)
    return area


def GetLineLength(p1, p2):
    length = math.pow((p1[0] - p2[0]), 2) + math.pow((p1[1] - p2[1]), 2)  # pow  次方
    length = math.sqrt(length)
    return length


def vec_angle(vec1_pt1, vec1_pt2, vec2_pt1, vec2_pt2):
    """
    Calculating angle between vect1 (pt1->pt2) and vect2 (pt1->pt2)
    An utils for def SI_calculate()
    """
    dx1 = vec1_pt1[0] - vec1_pt2[0]
    dy1 = vec1_pt1[1] - vec1_pt2[1]
    dx2 = vec2_pt1[0] - vec2_pt2[0]
    dy2 = vec2_pt1[1] - vec2_pt2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180 / math.pi)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180 / math.pi)
    if angle1 * angle2 >= 0:
        insideAngle = abs(angle1 - angle2)
    else:
        insideAngle = abs(angle1) + abs(angle2)
        if insideAngle > 180:
            insideAngle = 360 - insideAngle
    insideAngle = insideAngle % 180
    return insideAngle


def pts_distance(point1, point2):
    """
    Calculating Euclidean distance:
    point1 -> point2
    An utils for def SI_calculate()
    """
    distance = math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))
    return distance


def gamma(value, K, lamb):
    """
    Gamma correction method for reducing measurement error:
    NewValue = K*Value^lamb
    An utils for def SI_calculate()
    """
    newValue = K * math.pow(value, lamb)
    return newValue
