# -*- coding: utf-8 -*-
# @Time : 2022/8/21
# @Author : Miao Shen
# @File : test.py

import numpy as np



class User():
    """
    basic block
    """
    def __init__(self, name,phone):
        """

        :param in_channels:
        :param middle_channels:
        :param out_channels:
        """
        super().__init__()
        self.name=name
        self.phone=phone

    def out(self):
        """

        :return: information
        """
        print(str(self.name)+str(self.phone))


class Model():
    """
    test model
    """
    def __init__(self, name,num_class):
        """

        :param name:
        :param num_class:
        """

        super().__init__()

        self.name=name
        self.num_class=num_class

    def out(self):
        """

        :return: null
        """
        print(self.name)

