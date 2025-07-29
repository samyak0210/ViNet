# ViNet: Pushing the limits of Visual Modality for Audio-Visual Saliency Prediction

----
> ## 🚀 New and Improved ViNet++ Models Available!
>
> We’ve released updated ViNet models with significantly improved speed and performance in our latest paper: [ViNet++: Minimalistic Video Saliency Prediction via Efficient Decoder & Spatio Temporal Action Cues](https://ieeexplore.ieee.org/abstract/document/10888852) with Code and pre-trained checkpoints available at [ViNet++ Github Repository](https://github.com/ViNet-Saliency/vinet_v2)
----

This repository contains Pytorch Implementation of ViNet and AViNet.

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/avinet-diving-deep-into-audio-visual-saliency/video-saliency-detection-on-dhf1k)](https://paperswithcode.com/sota/video-saliency-detection-on-dhf1k?p=avinet-diving-deep-into-audio-visual-saliency)

## Cite
Please cite with the following Bibtex code:
```
@misc{jain2021vinet,
      title={ViNet: Pushing the limits of Visual Modality for Audio-Visual Saliency Prediction}, 
      author={Samyak Jain and Pradeep Yarlagadda and Shreyank Jyoti and Shyamgopal Karthik and Ramanathan Subramanian and Vineet Gandhi},
      year={2021},
      eprint={2012.06170},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
## Abstract

We propose the ViNet architecture for audio-visual saliency prediction. ViNet is a fully convolutional encoder-decoder architecture. The encoder uses visual features from a network trained for action recognition, and the decoder infers a saliency map via trilinear interpolation and 3D convolutions, combining features from multiple hierarchies. The overall architecture of ViNet is conceptually simple; it is causal and runs in real-time (60 fps). ViNet does not use audio as input and still outperforms the state-of-the-art audio-visual saliency prediction models on nine different datasets (three visual-only and six audio-visual datasets). ViNet also surpasses human performance on the CC, SIM and AUC metrics for the AVE dataset, and to our knowledge, it is the first network to do so. We also explore a variation of ViNet architecture by augmenting audio features into the decoder. To our surprise, upon sufficient training, the network becomes agnostic to the input audio and provides the same output irrespective of the input. Interestingly, we also observe similar behaviour in the previous state-of-the-art models \cite{tsiami2020stavis} for audio-visual saliency prediction. Our findings contrast with previous works on deep learning-based audio-visual saliency prediction, suggesting a clear avenue for future explorations incorporating audio in a more effective manner.

## Examples
We compare our model ViNet with UNISAL (previous state-of-the-art). Below are some examples of our model. The first section is original video, second is the ground-truth, third is our model's prediction and last is the UNISAL's prediction.

![](./extras/0601.gif)

![](./extras/0654.gif)

## Architecture
![](./extras/AViNet.jpg)

## Dataset
* DHF1K and UCF Sports dataset can be downloaded from this [link](https://drive.google.com/drive/folders/1sW0tf9RQMO4RR7SyKhU8Kmbm4jwkFGpQ).
* Hollywood-2 dataset can be downloaded from this [link](https://drive.google.com/file/d/1vfRKJloNSIczYEOVjB4zMK8r0k4VJuWk/view)
* The six audio-visual datasets - DIEM, AVAD, Coutrot-1&2, SumMe and ETMD can be downloaded from this [link](http://cvsp.cs.ntua.gr/research/stavis/data/). You can also run the following command to fetch all the six dataset and its components - 
```bash
$ bash fetch_data.sh
```


## Testing
Clone this repository and download the pretrained weights of AViNet and ViNet on multiple datasets from this [link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/samyak_j_research_iiit_ac_in/EXYq5WiSbh9Kq9R_n-Gr3yABRyKPSkxM7ROLg-zPDXV_qA?e=5AL7UU).

* ### ViNet
Run the code using 
```bash
$ python3 generate_result.py --path_indata path/to/test/frames --save_path path/to/results --file_weight path/to/saved/models
```
This will generate saliency maps for all frames in the directory and dump these maps into results directory. The directory structure should be 
```
└── Dataset  
    ├── Video-Number  
        ├── images  
```
* ### AViNet
Run the code using 
```bash
$ python3 generate_result_audio_visual.py --path_indata path/to/test/frames --save_path path/to/results --file_weight path/to/saved/models --use_sound True --split <split_number>
<split_number>: {1,2,3}
```
This will generate saliency maps for all frames in the directory and dump these maps into results directory. The directory structure should be 
```
└── Dataset  
    ├── video_frames  
        ├── <dataset_name>
            ├── Video-Name
                ├── frames
    ├── video_audio  
        ├── <dataset_name>
            ├── Video-Name
                ├── audio  
    ├── fold_lists
        ├── <dataset_file>.txt
```
Fold_lists consists of text file of video names and their corresponding fps in various splits. The directory structure is the same as the one generated by the fetch_data.sh file.

## Training
For training the model from scratch, download the pretrained weights of S3D from [here](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/samyak_j_research_iiit_ac_in/EYZ8Elhmc9tOmlVwnb41GEEBQNnmW31Q2mAwE8B9sFn7WA?e=co9Hvj) and place these weights in the same directory. Run the following command to train 

```bash
$ python3 train.py --train_path_data path/to/train/dataset --val_path_data  path/to/val/dataset --dataset <dataset_name> --use_sound <boolean_value>
<dataset_name> : {"DHF1KDataset", "SoundDataset", "Hollywood", "UCF"} 
```
In case of ViNet, the dataset directory structure should be
```
└── Dataset  
    ├── Video-Number  
        ├── images  
        |── maps
        └── fixations  
```

In case of AViNet, the dataset directory structure should be

```
└── Dataset  
    ├── video_frames  
        ├── <dataset_name>
            ├── Video-Name
                ├── frames
    ├── video_audio  
        ├── <dataset_name>
            ├── Video-Name
                ├── audio
    ├── annotations
        ├── <dataset_name>
            ├── Video-Name
                ├── <frame_id>.mat (fixations)
                ├── maps
                    ├── <frame_id>.jpg (ground truth saliency map)
    ├── fold_lists
        ├── <dataset_file>.txt
```

For training the ViNet with Hollywood-2 or UCF-Sports dataset, first train the model with DHF1K dataset or you can directly use our model trained on DHF1K, and finetune the model weights on aforementioned datasets.

Similarly for training the AViNet with DIEM, AVAD, Coutrot-1&2, ETMD and SumMe dataset, first load model with DHF1K trained weights and finetune the model weights on aforementioned datasets.

## Experiments

* ### Audio
For training the model, we provide argument to select the model between ViNet (Visual Net) and AViNet (Audio-Visual Net). Run the command - 
```bash
$ python3 train.py --use_sound <boolean_value> 
```

If you want to save the results of the generated map run the command - 
```bash
$ python3 generate_result_audio_visual.py --use_sound <boolean_value> --file_weight <path/to/file> --path_indata <path/to/data> 
```

* ### Multiple Audio-Visual Fusion 
You can select the corresponding fusion technique's model from the model.py file. By default it uses the model with Bilinear concatenation for fusing audio and visual cues. If you want to use Transformer-based fusion technique, call the model with name VideoAudioSaliencyTransformerFusionModel.

## Quantitative Results

* ### DHF1K
The results of our models on DHF1K test dataset can be viewed [here](https://mmcheng.net/videosal/) under the name ViNet. Comparison with other state-of-the-art saliency detection models 
![](./extras/DHF1K.png)

* ### Hollywood2
![](./extras/Hollywood.png)

* ### UCF
![](./extras/UCF.png)

* ### DIEM
We provide results on both our models - ViNet and AViNet on the DIEM dataset.
![](./extras/DIEM.png)

## Contact 
If any question, please contact samyak.j@research.iiit.ac.in, or pradeep.yarlagadda@students.iiit.ac.in , or use public issues section of this repository

## License 
This code is distributed under MIT LICENSE.
