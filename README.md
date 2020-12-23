# AViNet: Diving Deep into Audio-Visual Saliency Prediction

This repository contains Pytorch Implementation of ViNet and AViNet.

## Cite
Please cite with the following Bibtex code:
```
@article{jain2020avinet,
  title={AViNet: Diving Deep into Audio-Visual Saliency Prediction},
  author={Jain, Samyak and Yarlagadda, Pradeep and Subramanian, Ramanathan and Gandhi, Vineet},
  journal={arXiv preprint arXiv:2012.06170},
  year={2020}
}
```
## Abstract

We propose the AViNet architecture for audiovisual
saliency prediction. AViNet is a fully convolutional encoderdecoder architecture. The encoder combines visual features learned for action recognition, with audio embeddings learned via an aural network designed to classify
objects and scenes. The decoder infers a saliency map
via trilinear interpolation and 3D convolutions, combining hierarchical features. The overall architecture is conceptually simple, causal, and runs in real-time (60 fps).
AViNet outperforms the state-of-the-art on ten (seven audiovisual and three visual-only) datasets, while surpassing human performance on the CC, SIM and AUC metrics for the AVE dataset. Visual features maximally account
for saliency on existing datasets with audio only contributing to minor gains, except in specific contexts like social
events. Our work therefore motivates the need to curate
saliency datasets reflective of real-life, where both the visual and aural modalities complimentarily drive saliency

## Architecture
![](./extras/AViNet.jpg)

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
Fold_lists consists of text file of video names in various splits. 

## Training
For training the model from scratch, download the pretrained weights of S3D from [here](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/samyak_j_research_iiit_ac_in/EYZ8Elhmc9tOmlVwnb41GEEBQNnmW31Q2mAwE8B9sFn7WA?e=co9Hvj) and place these weights in the same directory. Run the following command to train 

```bash
$ python3 train.py --train_path_data path/to/train/dataset --val_path_data  path/to/val/dataset --dataset <dataset_name> --use_sound <boolean_value>
<dataset_name> : {"DHF1KDataset", "SoundDataset", "Hollywood"/"UCF"} 
```
The dataset directory structure should be - in case of ViNet 
```
└── Dataset  
    ├── Video-Number  
        ├── images  
        |── maps
        └── fixations  
```

The dataset directory structure should be - in case of AViNet

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
                ├── <frame_id>.mat
                ├── maps
                    ├── <frame_id>.jpg  
    ├── fold_lists
        ├── <dataset_file>.txt
```

For training the ViNet with Hollywood-2 or UCF-Sports dataset, first train the model with DHF1K dataset and finetune the model weights on aforementioned datasets.

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
You can select the corresponding fusion technique's model from the model.py file. 

## Quantitative Results

* ### DHF1K
The results of our models on SALICON test dataset can be viewed [here](https://mmcheng.net/videosal/) under the name ViNet. Comparison with other state-of-the-art saliency detection models 
![](./extras/DHF1K.png)

## Contact 
If any question, please contact samyak.j@research.iiit.ac.in, or pradeep.yarlagadda@students.iiit.ac.in , or use public issues section of this repository

## License 
This code is distributed under MIT LICENSE.