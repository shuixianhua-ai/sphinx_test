# -*- coding: utf-8 -*-
# @Author  : Miao Shen
# @Time    : 2022/8/21
# @File    : flood_extraction_demo.py



import argparse
from model.flood import FloodInundation
from preprocessing.dataprocess import Dataprocess,CutImage


def get_args():

    parse = argparse.ArgumentParser(description="flood inundation extraction",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument("-img1", "--sentinel1_pre", dest="sentinel1_pre", required=True,
                       type=str, default="data/Iran_s1_pre.tif", help="sentinel-1 image before disaster")
    parse.add_argument("-img2", "--sentinel2_pre", dest="sentinel2_pre", required=True,
                       type=str, default="data/Iran_s2_pre.tif", help="sentinel-2 image before disaster")
    parse.add_argument("-img3", "--sentinel1_post", dest="sentinel1_post", required=True,
                       type=str, default="data/Iran_s1_post.tif", help="sentinel-1 image after disaster")
    parse.add_argument("-img4", "--sentinel2_post", dest="sentinel2_post", required=True,
                       type=str, default="data/Iran_s2_post.tif", help="sentinel-2 image after disaster")

    parse.add_argument("-k1", "--threshold1", dest="threshold1", type=float, default=0.4,
                       help="the first threshold for decision-level data fusion")
    parse.add_argument("-k2", "--threshold2", dest="threshold2", type=float, default=0.95,
                       help="the second threshold for decision-level data fusion")
    parse.add_argument("-k3", "--threshold3", dest="threshold3", type=float, default=0.5,
                       help="the third threshold for decision-level data fusion")

    parse.add_argument("-s", "--use_sentinel2", dest="use_sentinel2", required=True,
                       type=bool, default=False, help="extract flood inundation areas use sentinel-1 or use the fusion of sentinel-1 and sentinel-2")

    parse.add_argument("-o", "--output", dest="output", required=True,
                       type=str, default="data/output.tif", help="output file path")


    return parse.parse_args()


if __name__ == '__main__':
    args = get_args()
    print("load parse succeed.")

    s1_pre_path = args.sentinel1_pre
    s2_pre_path = args.sentinel2_pre
    s1_post_path = args.sentinel1_post
    s2_post_path = args.sentinel2_post

    k1 = args.threshold1
    k2 = args.threshold2
    k3 = args.threshold3

    use_s2 = args.use_sentinel2

    output = args.output


    # cut the iamge into 512 pixels * 512 pixels
    CutImage_pre=CutImage(s1_path=s1_pre_path,s2_path=s2_pre_path)
    cutimage_pre=CutImage_pre.run()
    CutImage_post = CutImage(s1_path=s1_post_path, s2_path=s2_post_path)
    cutimage_post=CutImage.run()


    # make multi-channel data
    Dataprocess_pre = Dataprocess(data=cutimage_pre, use_s2=use_s2)
    data_pre = Dataprocess_pre.out()
    Dataprocess_post = Dataprocess(data=cutimage_pre, use_s2=use_s2)
    data_post = Dataprocess_post.out()


    input_list = ModelIOList()
    param_list = ModelIOList()
    output_list = ModelIOList()

    input1 = ModelIO()
    input1.name = 'input_pre'
    input1.data = data_pre
    input_list.append(input1)

    input2 = ModelIO()
    input2.name = 'input_post'
    input2.data = data_post
    input_list.append(input2)

    param1 = ModelIO()
    param1.name = 'use_s2'
    param1.data = use_s2
    param_list.append(param1)

    param2 = ModelIO()
    param2.name = 'threshold1'
    param2.data = k1
    param_list.append(param2)

    param3 = ModelIO()
    param3.name = 'threshold2'
    param3.data = k2
    param_list.append(param3)

    param4 = ModelIO()
    param4.name = 'threshold3'
    param4.data = k3
    param_list.append(param4)


    output1 = ModelIO()
    output1.name = 'output'
    output1.data = output
    output_list.append(output1)



    model = FloodInundation(data_pre=data_pre,data_post=data_post,use_s2=use_s2,k1=k1,k2=k2,k3=k3,output=output,image_path=s1_pre_path)
    print("flood inundation extraction method construct finished.")
    res=model.run()
    print("ready to output.")
    model.out(res)
    print("flood inundation extraction finished.")
