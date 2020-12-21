The code to compute the saliency metrics are used in DHF1K (https://github.com/wenguanwang/DHF1K), which is modified from the evaluation tool in http://saliency.mit.edu.

Note that the binary human fixation map (zero matrix with 1s at exact locations of fixations) is used as the fixationMap argument to most of the metrics. If, however, a continuous map is required (e.g. for CC and similarity computations), then the binary human fixation map is blurred. Specifically, we use the antonioGaussian.m low-pass filter with cut off frequency fc = 8 cycles per image, which is approximately equivalent to 1 degree of visual angle.

Note that to compute EMD, we use the FastEMD approximation provided by [Ofir Pele](http://www.ariel.ac.il/sites/ofirpele/fastemd/code/). Please download and compile the FastEMD code available at [http://www.ariel.ac.il/sites/ofirpele/fastemd/code/](http://www.ariel.ac.il/sites/ofirpele/fastemd/code/).

If you find our dataset is useful, please cite the following papers.

@InProceedings{Wang_2018_CVPR,
author = {Wang, Wenguan and Shen, Jianbing and Guo, Fang and Cheng, Ming-Ming and Borji, Ali},
title = {Revisiting Video Saliency: A Large-Scale Benchmark and a New Model},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition},
year = {2018}
}

@ARTICLE{Wang_2019_revisitingVS, 
author={W. {Wang} and J. {Shen} and J. {Xie} and M. {Cheng} and H. {Ling} and A. {Borji}}, 
journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
title={Revisiting Video Saliency Prediction in the Deep Learning Era}, 
year={2019}, 
}