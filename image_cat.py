import cv2
import numpy as np
import os


# 分割边缘 edge=1表示粗糙度为1
def edge_crop(edge=1):
   if edge == 1:
      filename = './hand_input/edge_big1'
      savename = './data/edge_small1'
   elif edge == 2:
      filename = './hand_input/edge_big2'
      savename = './data/edge_small2'
   elif edge == 3:
      filename = './hand_input/edge_big3'
      savename = './data/edge_small3'
   else:
      print('您输入的粗糙度有误，不属于123')

   data = os.listdir(filename)

   count = 0
   for j in range(len(data)):
      j = j + 1
      img_none = cv2.imread('%s/edge%d.jpg' % (filename, j))   # 读取图片


      weight = img_none.shape[1]   # 图片的宽
      high = img_none.shape[0]   # 图片的高
      for x in range(weight // 64):
            res = img_none[0:64, x*64:x*64+64]
            cv2.imwrite('%s/%s' % (savename, str(count) + '.jpg'), res)  # 输出裁剪后的图片
            count = count + 1   # 分割边缘的

# 分割浆体
def ce_crop():

   data = os.listdir('./hand_input/cement_big')

   count = 0
   for j in range(len(data)):
      j = j + 1
      img_none = cv2.imread('./hand_input/cement_big/ce%d.jpg' % j)   # 读取图片

      weight = img_none.shape[1]   # 图片的宽
      high = img_none.shape[0]   # 图片的高
      for x in range(weight // 64):
         for y in range(high // 64):
            res = img_none[y*64:y*64+64, x*64:x*64+64]
            cv2.imwrite('./data/ce_black/%d.jpg' % count, res)  # 输出裁剪后的图片
            count = count + 1
      for x in range(img_none.shape[0]):
         for y in range(img_none.shape[1]):
            if img_none[x][y][0] < 128:
               img_none[x][y] = [255, 0, 0]   # 蓝色
      count = 0
      for x in range(weight // 64):
         for y in range(high // 64):
            res = img_none[y*64:y*64+64, x*64:x*64+64]
            cv2.imwrite('./data/cement_small/%d.jpg' % count, res)  # 输出裁剪后的图片
            count = count + 1

# 分割骨料
def agg_crop():

   data = os.listdir('./hand_input/aggregate_big')

   count = 0
   for j in range(len(data)):
      j = j + 1
      img_none = cv2.imread('./hand_input/aggregate_big/agg%d.jpg' % j)   # 读取图片

      weight = img_none.shape[1]   # 图片的宽
      high = img_none.shape[0]   # 图片的高
      for x in range(weight // 64):
         for y in range(high // 64):
            res = img_none[y*64:y*64+64, x*64:x*64+64]
            cv2.imwrite('./data/agg_black/%d.jpg' % count, res)  # 输出裁剪后的图片
            count = count + 1

      for x in range(img_none.shape[0]):
         for y in range(img_none.shape[1]):
            if img_none[x][y][0] > 128:
               img_none[x][y] = [0, 0, 255]   # 红色

      count = 0
      for x in range(weight // 64):
          for y in range(high // 64):
              res = img_none[y * 64:y * 64 + 64, x * 64:x * 64 + 64]
              cv2.imwrite('./data/aggregate_small/%d.jpg' % count, res)  # 输出裁剪后的图片
              count = count + 1

def cat(num=1, resolution=5120, agg_resolution=1024, rough=1, rotate=0):
    if rotate != 0:
        original_resolution = resolution
        resolution = int(1.5 * resolution)
    data1 = os.listdir('./data/gen_ce')
    data2 = os.listdir('./data/gen_agg')
    for epoch in range(num):
        epoch = epoch + 1
        # 重构浆体分辨率=resolution
        combine = cv2.imread('./data/gen_ce/%d.jpg' % int(np.random.randint(0, len(data1))))
        for i in range((resolution // 64) - 1):
            b = cv2.imread('./data/gen_ce/%d.jpg' % int(np.random.randint(0, len(data1))))  # 随机抽一块
            combine = cv2.hconcat([combine, b])

        for i in range((resolution // 64) - 1):
            combine1 = cv2.imread('./data/gen_ce/%d.jpg' % int(np.random.randint(0, len(data1))))  # 随机抽一块
            for i in range((resolution // 64) - 1):
                b = cv2.imread('./data/gen_ce/%d.jpg' % int(np.random.randint(0, len(data1))))  # 随机抽一块
                combine1 = cv2.hconcat([combine1, b])
            combine = cv2.vconcat([combine, combine1])

        # 重构骨块分辨率=agg_resolution
        agg_combine = cv2.imread('./data/gen_agg/%d.jpg' % int(np.random.randint(0, len(data2))))
        for i in range((agg_resolution // 64) - 1):
            b = cv2.imread('./data/gen_agg/%d.jpg' % int(np.random.randint(0, len(data2))))  # 随机抽一块
            agg_combine = cv2.hconcat([agg_combine, b])

        for i in range((agg_resolution // 64) - 1):
            agg_combine1 = cv2.imread('./data/gen_agg/%d.jpg' % int(np.random.randint(0, len(data2))))  # 随机抽一块
            for i in range((agg_resolution // 64) - 1):
                b = cv2.imread('./data/gen_agg/%d.jpg' % int(np.random.randint(0, len(data2))))  # 随机抽一块
                agg_combine1 = cv2.hconcat([agg_combine1, b])
            agg_combine = cv2.vconcat([agg_combine, agg_combine1])

        # 浆体和骨块拼接
        x1 = resolution // 2 - agg_resolution // 2 - 1
        x2 = resolution // 2 + agg_resolution // 2 - 1
        x3 = resolution // 2 - agg_resolution // 2 - 1
        x4 = resolution // 2 + agg_resolution // 2 - 1
        combine[x1:x2, x3:x4] = agg_combine

        # 选择粗糙度
        if rough == 1:
            angle = './hand_input/angle1'
            edge = './data/edge_small1'
        elif rough == 2:
            angle = './hand_input/angle2'
            edge = './data/edge_small2'
        elif rough == 3:
            angle = './hand_input/angle3'
            edge = './data/edge_small3'
        else:
            print('您输入的粗糙度不属于1，2，3之内')
        data1 = os.listdir('%s' % edge)
        # 添加四个角
        a = cv2.imread('%s/angle1.jpg' % angle)
        b = cv2.rotate(a, cv2.ROTATE_90_CLOCKWISE)  # 旋转90度
        c = cv2.rotate(a, cv2.ROTATE_180)  # 旋转180度
        d = cv2.rotate(a, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 旋转270度

        combine[x1:x1 + 128, x1:x1 + 128] = a    #  [y轴, x轴]
        combine[x1:x1 + 128, x2 - 128:x2] = b
        combine[x2 - 128:x2, x2 - 128:x2] = c
        combine[x2 - 128:x2, x1:x1 + 128] = d

        # 添加边缘
        combine1 = cv2.imread('%s/%d.jpg' % (edge, int(np.random.randint(1, len(data1)))))
        for i in range(((agg_resolution - 256) // 64) - 1):
            a_ = cv2.imread('%s/%d.jpg' % (edge, int(np.random.randint(1, len(data1)))))
            combine1 = cv2.hconcat([combine1, a_])


        combine2 = cv2.imread('%s/%d.jpg' % (edge, int(np.random.randint(1, len(data1)))))
        combine2 = cv2.rotate(combine2, cv2.ROTATE_90_CLOCKWISE)
        for i in range(((agg_resolution - 256) // 64) - 1):
            b_ = cv2.imread('%s/%d.jpg' % (edge, int(np.random.randint(1, len(data1)))))
            b_ = cv2.rotate(b_, cv2.ROTATE_90_CLOCKWISE)
            combine2 = cv2.vconcat([combine2, b_])


        combine3 = cv2.imread('%s/%d.jpg' % (edge, int(np.random.randint(1, len(data1)))))
        combine3 = cv2.rotate(combine3, cv2.ROTATE_180)
        for i in range(((agg_resolution - 256) // 64) - 1):
            c_ = cv2.imread('%s/%d.jpg' % (edge, int(np.random.randint(1, len(data1)))))
            c_ = cv2.rotate(c_, cv2.ROTATE_180)
            combine3 = cv2.hconcat([combine3, c_])

        combine4 = cv2.imread('%s/%d.jpg' % (edge, int(np.random.randint(1, len(data1)))))
        combine4 = cv2.rotate(combine4, cv2.ROTATE_90_COUNTERCLOCKWISE)
        for i in range(((agg_resolution - 256) // 64) - 1):
            d_ = cv2.imread('%s/%d.jpg' % (edge, int(np.random.randint(1, len(data1)))))
            d_ = cv2.rotate(d_, cv2.ROTATE_90_COUNTERCLOCKWISE)
            combine4 = cv2.vconcat([combine4, d_])

        # 覆盖边缘
        combine[x1:x1 + 64, x1 + 128:x2 - 128] = combine1  # [y轴, x轴]
        combine[x1 + 128:x2 - 128, x2 - 64:x2] = combine2
        combine[x2 - 64:x2, x1 + 128:x2 - 128] = combine3
        combine[x1 + 128:x2 - 128, x1:x1 + 64] = combine4

        if rotate != 0:
            cx = cy = resolution // 2
            M = cv2.getRotationMatrix2D((cx, cy), rotate, 1.0)

            nW = nH = int(resolution * 1.5)

            M[0, 2] += (nW / 2) - cx
            M[1, 2] += (nH / 2) - cy

            combine = cv2.warpAffine(combine, M, (nW, nH))
            x1 = nW // 2 - original_resolution // 2
            x2 = x1 + original_resolution
            combine = combine[x1:x2, x1:x2]
        if rotate != 0:
            cv2.imwrite(
                './result/rough%d/%d_%d_%d_%d.jpg' % (rough, epoch, original_resolution, agg_resolution, rotate),
                combine)
        else:
            cv2.imwrite(
                './result/rough%d/%d_%d_%d_%d.jpg' % (rough, epoch, resolution, agg_resolution, rotate),
                combine)


if __name__ == '__main__':

    # 是否需要裁剪原始图像？ 需要请设置为True，不需要请设置为False，一般裁剪过一次就无需再裁剪
    crop = True
    # 如果选择裁剪原始图像，请设置骨料粗糙度
    crop_rough = 1

    if not crop:  # 裁剪图像时不会合成图像，不裁剪图像时会合成图像
        num = 1   # num为生成图像的数量，每张都是随机生成，内容均不相同
        resolution = 5120  # 为整体图像分辨率`
        agg_resolution = 1024  # 为骨料图像分辨率  15mm 3072 / 10mm 2048 / 5mm 1024
        rough = 1  # 粗糙度，可选1未打磨,2粗打磨,3细打磨
        rotate = 45  # 角度，可选0-360，为逆时针旋转

    # 一键生成程序，一共生成9*3=27张图像，每种粗糙度有9张图像，三种角度*三种骨料占比=9张图像
    # 如要开启，请输入True
    moss = False

# ————————以下程序自动执行，请勿修改————————
    if crop:
        edge_crop(crop_rough)
        ce_crop()
        agg_crop()
    if moss:
        for rough in range(3):
            rough = rough + 1
            for rotate in (0, 45, 60):
                for agg_resolution in (1024, 2048, 3072):
                    cat(num, resolution, agg_resolution, rough, rotate)
    else:
        cat(num, resolution, agg_resolution, rough, rotate)





















