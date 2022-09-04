# -*- coding: utf-8 -*-
# @Time : 2022/8/21
# @Author : Miao Shen
# @File : Flood Inundation Extraction

# from pydde.core.model import BaseProcedureModel
# import numpy as np
# import torch
# import torch.nn as nn
# import rasterio




class FloodInundation():

    """
    FloodInundation is used to achieve a decision-level data fusion method. The data fusion
    method combines three different UNet++ models trained by Sentinel-1 and Sentinel-2 data,
    which produces more accurate flood inundated areas.

    -------------------
    :param data_pre: multi-channel data before disaster.
    :param data_post: multi-channel data before disaster.
    :param use_s2: If FALSE, the method will extract flood inundated areas only by Sentinel-1 images.If
                TRUE, the method will fuse Sentinel-1 and Sentinel-2 images by decision-level data
                fusion method.
    :param k1: The first threshold for decision-level data fusion method. In the first step of
                the decision-level data fusion method, the model good at removing non-water areas
                is used to calculate the probability of each pixel classified as water. The pixels
                with lower than k1 is defined as non-water.
    :param k2: The second threshold for decision-level data fusion method. In the second step of
                the decision-level data fusion method, the model good at distinguish water in cloud-free
                area is used to calculate the probability of each pixel classified as water. The pixels
                with higher than k2 is defined as water.
    :param k3: The third threshold for decision-level data fusion method. In the third step of
                the decision-level data fusion method, the model good at distinguish water in cloud
                area is used to extract flood inundation areas.
    :param output: path of output image
    :param image_path: path of original image(use to get profile)
    :param size: size of images

    """


    def __init__(self, data_pre, data_post, use_s2, k1, k2, k3,output,image_path,size=512):

        super().__init__()
        image = rasterio.open(image_path)
        self.data_pre = data_pre
        self.data_post = data_post
        self.use_s2 = use_s2
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.output=output
        self.profile = image.profile
        self.height = image.height
        self.width = image.width
        self.size=size


        self.model_s1 = torch.load("Perus_s1_0_0.4401586055755615.cp")
        self.model_band12 = torch.load("Perus_s2_13_3_0_0.5925423502922058.cp")
        self.model_df = torch.load("Perus_s1+s2a+ndm__3_7_0.6533569693565369.cp")

    def extraction(self):
        """
        extract flood inundated areas
        ---------------------------------
        :return: results of flood inundated areas in size of 512*512
        """


        img = []

        with torch.no_grad():
            print(len(self.test_data_post))
            for i in range(0, len(self.test_data_post)):
                if self.use_s2 == True:
                    outputs_pre = self.data_fusion(self.data_pre)
                    outputs_post = self.data_fusion(self.data_post)
                else:
                    outputs_pre = self.data_fusion_s1(self.data_pre)
                    outputs_post = self.data_fusion_s1(self.data_post)

                outputs_pre = outputs_pre.flatten()
                outputs_post = outputs_post.flatten()

                output = torch.sub(outputs_post, outputs_pre)
                zero_out = torch.zeros_like(output)
                output = torch.where(output < 0, zero_out, output)
                output = output.cpu().numpy()
                output = np.array(output)
                output = output.reshape((4, 256, 256))

                temp_out = np.zeros(shape=(512, 512), dtype=np.uint8)
                temp_out[0:256, 0:256] = output[0]
                temp_out[256:512, 0:256] = output[2]
                temp_out[0:256, 256:512] = output[1]
                temp_out[256:512, 256:512] = output[3]
                img.append(temp_out)

        return img

    def out(self,img):
        """
        output the flood inundation map in study area

        """

        x = int(np.ceil(self.height / self.size)) - 1
        y = int(np.ceil(self.width / self.size)) - 1
        out = np.zeros(shape=(1, self.height, self.width), dtype=np.uint8)

        for i in range(0, x + 1):
            for j in range(0, y + 1):
                print(i)
                print(j)
                if j == y and i == x:
                    out[0][i * 512:self.height, j * 512:self.width] = img[i * (y + 1) + j][0:(self.height - x * 512),
                                                            0:(self.width - y * 512)]
                    continue
                if j == y:
                    out[0][i * 512:(i + 1) * 512, j * 512:self.width] = img[i * (y + 1) + j][:, 0:(self.width - y * 512)]
                    continue
                if i == x:
                    out[0][i * 512:self.height, j * 512:(j + 1) * 512] = img[i * (y + 1) + j][0:(self.height - x * 512), :]
                    continue
                a = img[i * (y + 1) + j]
                out[0][i * 512:(i + 1) * 512, j * 512:(j + 1) * 512] = a

        # print(self.profile)

        with rasterio.open(self.output, mode='w', **self.profile) as dst:
            dst.write(out)

    def data_fusion_s1(self,test_data):
        """
        data fusion method for Sentinel-1
        ---------------------------------

        :param test_data: multi-channel data
        :rtype: numpy array
        :return: result extracted by Sentinel-1 data
        """

        images_s1 = test_data

        model_s1 = self.model_s1.eval()
        model_s1 = model_s1.cuda()

        out_s1 = model_s1(images_s1.cuda())

        pro_s1 = nn.functional.softmax(out_s1, dim=1)
        pro_s1 = pro_s1[:, 1, :, :]

        # fusion
        outputs_1 = torch.where(pro_s1 >= 0.5, 1, 0)

        return outputs_1

    def data_fusion(self,test_data):
        """
        decision-level data fusion method for combining  Sentinel-1 and Sentinel-2

        step 1 : remove non-water

        step 2 : extract water in non-cloud area

        step 3 : extract water in cloud area

        :param test_data: multi-channel data
        :return: result extracted by Sentinel-1 and Sentinel-2 data
        """


        model_s1 = self.model_s1.eval()
        model_s1 = model_s1.cuda()
        model_band12 = self.model_band12.eval()
        model_band12 = model_band12.cuda()
        model_df = self.model_df.eval()
        model_df = model_df.cuda()

        (img1,img2,img3)=test_data
        images_s1,images_band12,images_df=img1.copy(),img2.copy(),img3.copy()


        out_band12 = model_band12(images_band12.cuda())
        out_s1 = model_s1(images_s1.cuda())
        out_df = model_df(images_df.cuda())

        pro_band12 = nn.functional.softmax(out_band12, dim=1)
        pro_band12 = pro_band12[:, 1, :, :]
        pro_s1 = nn.functional.softmax(out_s1, dim=1)
        pro_s1 = pro_s1[:, 1, :, :]
        pro_df = nn.functional.softmax(out_df, dim=1)
        pro_df = pro_df[:, 1, :, :]

        # step 1 : remove non-water
        outputs_1 = torch.add(pro_band12 * 0.5, pro_s1 * 0.5)
        outputs_1[outputs_1 < 0.1] = 0
        outputs_1[outputs_1 >= 0.1] = 2

        # step 2 : extract water in non-cloud area
        outputs_2 = torch.zeros_like(pro_band12)
        outputs_2[pro_df < 0.02] = 0
        outputs_2[pro_df >= 0.02] = 2
        outputs_2[pro_df >= 0.95] = 1

        # step 3 : extract water in cloud area
        outputs_3 = torch.add(pro_df * 0.5, pro_s1 * 0.5)
        outputs_3[outputs_3 >= 0.5] = 1
        outputs_3[outputs_3 < 0.5] = 0

        # fusion
        outputs_1 = torch.where(outputs_1 >= 2, outputs_2, outputs_1)
        outputs_1 = torch.where(outputs_1 >= 2, outputs_3, outputs_1)

        return outputs_1




