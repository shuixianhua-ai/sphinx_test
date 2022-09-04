# -*- coding: utf-8 -*-
# @Time : 2022/8/21
# @Author : Miao Shen
# @File : cutimage.py


# import numpy as np
# from torchvision import transforms
# from PIL import Image
# import torchvision.transforms.functional as F
# import torch
# import logging
# from osgeo import gdal


class CutImage():

    """
    cut image


    :param s1_path: path of Sentinel-1 image.
    :param s2_path: path of Sentinel-2 image.
    :param SIZE: size of Sentinel-1 and Sentinel-2 images

            .. note:: To improve the efficient of extraction, the original Sentinel-1 images and Sentinel-2 images will be cut into 512 pixels * 512 pixels.
    :param s1_CHANNEL: number of Sentinel-1 channels
    :param s2_CHANNEL: number of Sentinel-2 channels
    :type s1_path: str
    :type s2_path: str
    """

    def __init__(self, s1_path, s2_path, SIZE=512,s1_CHANNEL=2,s2_CHANNEL=14):

        super().__init__()
        self.s1_path = s1_path
        self.s2_path = s2_path
        self.size = SIZE
        self.s1_channel=s1_CHANNEL
        self.s2_channel=s2_CHANNEL

    def run(self):
        """
        Get the result of image cutting


        :return: a group of Sentinel-1 and Sentinel-2 images in certain size
        """


        logging.info('cut images')

        s1 = cut(self.s1_path,self.size,self.s1_channel)
        s2 = cut(self.s2_path, self.size,self.s2_channel)

        flood_data = []
        for i in range(0, len(s1)):
            arr_s1 = s1[i].reshape(self.s1_channel, self.size, self.size)
            arr_s2 = s2[i].reshape(self.s2_channel, self.size, self.size)[1:, :, :]
            arr_s1 = np.clip(arr_s1, -50, 1)
            arr_s1 = (arr_s1 + 50) / 51
            flood_data.append((arr_s1, arr_s2))

        logging.info('cut images finish')
        return flood_data

def cut(in_file, size,channel):
    """
    cut images


    :param in_file: path of source image
    :param size: size of image
    :param channel: number of image's channel
    :return: a group of images in certain size
    :rtype: list
    """


    data = []
    image = gdal.Open(in_file).ReadAsArray()

    cut_factor_row = int(np.ceil(image.shape[1] / size))  # 0
    cut_factor_clo = int(np.ceil(image.shape[2] / size))  # 1

    temp = np.zeros((channel, size, size), dtype=float)
    temp_image = np.zeros_like(temp)
    num = 0

    for i in range(cut_factor_row):
        for j in range(cut_factor_clo):
            start_x = 0
            end_x = 0
            start_y = 0
            end_y = 0
            if i == cut_factor_row - 1:
                start_x = int(np.rint(i * size))
                end_x = image.shape[1]

            else:
                start_x = int(np.rint(i * size))
                end_x = int(np.rint((i + 1) * size))

            if j == cut_factor_clo - 1:
                start_y = int(np.rint(j * size))
                end_y = image.shape[2]

            else:
                start_y = int(np.rint(j * size))
                end_y = int(np.rint((j + 1) * size))

            temp_image[:, 0:(end_x - start_x), 0:(end_y - start_y)] = image[:, start_x:end_x, start_y:end_y]

            # 本地测试出现影像边缘异常，需要进行替换
            for k in range(0, len(temp_image)):
                s21, s22 = np.where(temp_image[k] < 0)
                # print(s21)
                for m in range(0, len(s21)):
                    temp_image[k][s21[m]][s22[m]] = 0
            data.append(temp_image)

    return data



class Dataprocess():
    """
    make multi-channel data


    :param data: the array of input data
    :param use_s2: whether to use Sentinel-2 images
    :type data: numpy array
    """

    def __init__(self, data,use_s2):
        """


        """

        super().__init__()
        self.data = data
        self.use_s2 = use_s2
        self.nor = [
            [1612.2641794179053, 694.6404158569574],
            [1379.8899556061613, 734.589213934987],
            [1344.4295633683826, 731.6118897277566],
            [1195.157229000143, 860.603745394514],
            [1439.168369746529, 771.3569863637912],
            [2344.2498120705645, 921.6300590130161],
            [2796.473722876989, 1088.0256714514674],
            [2578.4108992777597, 1029.246558060433],
            [3023.817505678254, 1205.1064480965915],
            [476.7287418382585, 331.6878880293502],
            [59.24111403905757, 130.40242222578226],
            [1989.1945548720423, 993.7071664926801],
            [1152.4886461779677, 768.8907975412457],
            [-0.2938129263276281, 0.21578320121968173],
            [-0.36928344017880277, 0.19538602918264955],
            [-6393.00646782349, 0.19538602918264955],
            [-2398.566478742078, 0.19538602918264955]
        ]

    def out(self):
        """
        output multi-channel data


        :return: multi-channel data
        :rtype: numpy array
        """

        data_s1 = self.processTestIm_s1()
        if self.use_s2 == True:
            data_band12= self.processTestIm_s1()
            data_df=self.processTestIm_df()
            return (data_s1,data_band12,data_df)
        else:
            return data_s1
            
           
        return flood_data

    def processTestIm_df(self):
        """
        make multi-channel data with Sentinel-1 bands, Sentinel-2 bands and water indices

                """
        norm = transforms.Normalize([0.6851,
                                     0.5235,
                                     self.nor[0][0],
                                     self.nor[1][0],
                                     self.nor[2][0],
                                     self.nor[3][0],
                                     self.nor[4][0],
                                     self.nor[5][0],
                                     self.nor[6][0],
                                     self.nor[7][0],
                                     self.nor[8][0],
                                     self.nor[9][0],
                                     self.nor[10][0],
                                     self.nor[11][0],
                                     self.nor[12][0],
                                     self.nor[13][0],
                                     self.nor[14][0]

                                     ],
                                    [0.0820,
                                     0.1102,
                                     self.nor[0][1],
                                     self.nor[1][1],
                                     self.nor[2][1],
                                     self.nor[3][1],
                                     self.nor[4][1],
                                     self.nor[5][1],
                                     self.nor[6][1],
                                     self.nor[7][1],
                                     self.nor[8][1],
                                     self.nor[9][1],
                                     self.nor[10][1],
                                     self.nor[11][1],
                                     self.nor[12][1],
                                     self.nor[13][1],
                                     self.nor[14][1]]
                                    )

        (sen1, sen2) = self.data

        s1, s2 = sen1.copy(), sen2.copy()

        # convert to PIL for easier transforms
        band_list = []
        for i in range(0, len(s1)):
            band = Image.fromarray(s2[i]).resize((512, 512))
            band_list.append(band)
        for i in range(0, len(s2)):
            band = Image.fromarray(s2[i]).resize((512, 512))
            band_list.append(band)

        ndw = (s2[2] - s2[7]) / (s2[2] + s2[7])
        ndw[np.isnan(ndw)] = 0
        ndwi = Image.fromarray(ndw).resize((512, 512))
        band_list.append(ndwi)

        ndm = (3.0 * s2[2] - s2[1] + 2.0 * s2[3] - 5.0 * s2[7]) / (3.0 * s2[2] + s2[1] + 2 * s2[3] + 5.0 * s2[7])
        ndm[np.isnan(ndm)] = 0
        ndmbwi = Image.fromarray(ndm).resize((512, 512))
        band_list.append(ndmbwi)

        bands_list = []
        for i in range(0, len(band_list)):
            bands = [F.crop(band_list[i], 0, 0, 256, 256), F.crop(band_list[i], 0, 256, 256, 256),
                     F.crop(band_list[i], 256, 0, 256, 256), F.crop(band_list[i], 256, 256, 256, 256)]
            bands_list.append(bands)

        ims = [torch.stack((transforms.ToTensor()(x1).squeeze(),
                            transforms.ToTensor()(x2).squeeze(),
                            transforms.ToTensor()(x3).squeeze(),
                            transforms.ToTensor()(x4).squeeze(),
                            transforms.ToTensor()(x5).squeeze(),
                            transforms.ToTensor()(x6).squeeze(),
                            transforms.ToTensor()(x7).squeeze(),
                            transforms.ToTensor()(x8).squeeze(),
                            transforms.ToTensor()(x9).squeeze(),
                            transforms.ToTensor()(x10).squeeze(),
                            transforms.ToTensor()(x11).squeeze(),
                            transforms.ToTensor()(x12).squeeze(),
                            transforms.ToTensor()(x13).squeeze(),
                            transforms.ToTensor()(x14).squeeze(),
                            transforms.ToTensor()(x15).squeeze(),
                            transforms.ToTensor()(x16).squeeze(),
                            transforms.ToTensor()(x17).squeeze()
                            ))
               for (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17) in
               zip(bands_list[0], bands_list[1], bands_list[2], bands_list[3], bands_list[4], bands_list[5],
                   bands_list[6],
                   bands_list[7], bands_list[8], bands_list[9], bands_list[10], bands_list[11], bands_list[12],
                   bands_list[13], bands_list[14], bands_list[15], bands_list[16])]

        ims = [norm(im) for im in ims]
        ims = torch.stack(ims)

        return ims

    def processTestIm_s1(self):
        """
        make multi-channel data with Sentinel-1 bands
        """
        norm = transforms.Normalize([0.6851, 0.5235], [0.0820, 0.1102])
        (sen1, sen2) = self.data
        im = sen1.copy()

        # convert to PIL for easier transforms
        vv = Image.fromarray(im[0]).resize((512, 512))
        vh = Image.fromarray(im[1]).resize((512, 512))

        vvs = [F.crop(vv, 0, 0, 256, 256), F.crop(vv, 0, 256, 256, 256),
               F.crop(vv, 256, 0, 256, 256), F.crop(vv, 256, 256, 256, 256)]
        vhs = [F.crop(vh, 0, 0, 256, 256), F.crop(vh, 0, 256, 256, 256),
               F.crop(vh, 256, 0, 256, 256), F.crop(vh, 256, 256, 256, 256)]

        ims = [torch.stack((transforms.ToTensor()(x1).squeeze(),
                            transforms.ToTensor()(x2).squeeze()))
               for (x1, x2) in zip(vvs, vhs)]

        ims = [norm(im) for im in ims]
        ims = torch.stack(ims)

        return ims

    def processTestIm_band12(self):
        """
                make multi-channel data with Sentinel-1 bands and Band12 from Sentinel-2
                """
        (sen1, sen2) = self.data
        s1, s2 = sen1.copy(), sen2.copy()
        norm = transforms.Normalize([0.6851, 0.5235, self.nor[12][0]], [0.0820, 0.1102, self.nor[12][1]])

        # convert to PIL for easier transforms
        vv = Image.fromarray(s1[0]).resize((512, 512))
        vh = Image.fromarray(s1[1]).resize((512, 512))
        band12 = Image.fromarray(s2[12]).resize((512, 512))

        vvs = [F.crop(vv, 0, 0, 256, 256), F.crop(vv, 0, 256, 256, 256),
               F.crop(vv, 256, 0, 256, 256), F.crop(vv, 256, 256, 256, 256)]
        vhs = [F.crop(vh, 0, 0, 256, 256), F.crop(vh, 0, 256, 256, 256),
               F.crop(vh, 256, 0, 256, 256), F.crop(vh, 256, 256, 256, 256)]
        band12s = [F.crop(band12, 0, 0, 256, 256), F.crop(band12, 0, 256, 256, 256),
                   F.crop(band12, 256, 0, 256, 256), F.crop(band12, 256, 256, 256, 256)]

        ims = [torch.stack((transforms.ToTensor()(x1).squeeze(),
                            transforms.ToTensor()(x2).squeeze(),
                            transforms.ToTensor()(x3).squeeze()))
               for (x1, x2, x3) in zip(vvs, vhs, band12s)]

        ims = [norm(im) for im in ims]
        ims = torch.stack(ims)

        return ims





