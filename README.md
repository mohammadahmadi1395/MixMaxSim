
  # MixMaxSim: Mixture of MaxMax and Similarity 📝  
  MixMaxSim project is mainly maintained By [Sayed Mohammad Ahmadi](mailto:ahmadi.mohammad2008@gmail.com) and [Ruhollah Dianat](mailto:ruhollah.dianat@gamil.com). </br></br>
  This project implements a framework in face identification which uses divide and conqure idea and add an intelligent post-processing to predict better.</br></br>
  This implementation is related to a manuscript which is under review in [IET Computer Vision Journal](https://ietresearch.onlinelibrary.wiley.com/journal/17519640) and is under the divide and conqure approach, such as [Independent Softmax Model](https://dl.acm.org/doi/abs/10.1145/2964284.2984060) paper and improves it by two modifications: using clustering instead of random distribution and using intelligent post-processing for combination of submodels. </br></br>
  This framework improves speed and memory and accuracy in comparison to single models.</br></br>
  This project is independent of base models and dataset. </br></br>
  Every one can use this framework for his own dataset or backbone.</br>

  ## Get Started 🚀  
  If you want implement the project in your machine, you need to have dataset, prepare the data and then implement the project.
  
  ## Requiremnts
  ``` pip install -r requiremnts.txt```

  ### Download datasets
  We used three datasets. You can download them or use your own dataset:
  1. Glint360k: magnet:?xt=urn:btih:E5F46EE502B9E76DA8CC3A0E4F7C17E4000C7B1E&dn=glint360k
  2. MS-Celeb-1M: Torrent file is located in: ./data/MS-Celeb-1M/MS-Celeb-1M-9e67eb7cc23c9417f39778a8e06cca5e26196a97.torrent
  3. VGGFace2: magnet:?xt=urn:btih:535113b8395832f09121bc53ac85d7bc8ef6fa5b&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php&tr=udp%3A%2F%2Ftracker.coppersurfer.tk%3A6969&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce

  ### Prepare data
  <b> IMPORTANT: The following steps are done for all datasets and for instance you can see data preparation for ms-celeb-1m dataset [here](./notebooks/ms1m_data_preparation.ipynb).</b> </br>
  Each dataset will be located in "./data/dataset_name/" </br>
  And includes three directories: train, test, val </br>
  Each directory includes directories with identity names. </br>
  For fair evaluation, for each identity, choose randomly 20 images for train, 5 for test, and 5 for validation. </br>
  For example:</br>
  ```
  ├── glint360k/
  │   ├── train/
  │   │   ├── id_1/
  │   │   │    ├── custom_name_1_1.jpg
  │   │   │    ├── custom_name_1_2.jpg
  │   │   │    ├── ...
  │   │   │    └── custom_name_1_20.jpg
  │   │   ├── id_2/
  │   │   │    ├── custom_name_2_1.jpg
  │   │   │    ├── custom_name_2_2.jpg
  │   │   │    ├── ...
  │   │   │    └── custom_name_2_20.jpg
  │   │   ├── ...
  │   │   │    ├── ...
  │   │   │    ├── ...
  │   │   │    ├── ...
  │   │   │    └── ...
  │   │   ├── id_360000/
  │   │   │    ├── custom_name_360000_1.jpg
  │   │   │    ├── custom_name_360000_2.jpg
  │   │   │    ├── ...
  │   │   │    └── custom_name_360000_20.jpg
  │   ├── val/
  │   │   ├── id_1/
  │   │   │    ├── custom_name_1_21.jpg
  │   │   │    ├── custom_name_1_22.jpg
  │   │   │    ├── ...
  │   │   │    └── custom_name_1_25.jpg
  │   │   ├── id_2/
  │   │   │    ├── custom_name_2_21.jpg
  │   │   │    ├── custom_name_2_22.jpg
  │   │   │    ├── ...
  │   │   │    └── custom_name_2_25.jpg
  │   │   ├── ...
  │   │   │    ├── ...
  │   │   │    ├── ...
  │   │   │    ├── ...
  │   │   │    └── ...
  │   │   ├── id_360000/
  │   │   │    ├── custom_name_360000_21.jpg
  │   │   │    ├── custom_name_360000_22.jpg
  │   │   │    ├── ...
  │   │   │    └── custom_name_360000_25.jpg
  │   ├── test/
  │   │   ├── id_1/
  │   │   │    ├── custom_name_1_26.jpg
  │   │   │    ├── custom_name_1_27.jpg
  │   │   │    ├── ...
  │   │   │    └── custom_name_1_30.jpg
  │   │   ├── id_2/
  │   │   │    ├── custom_name_2_26.jpg
  │   │   │    ├── custom_name_2_27.jpg
  │   │   │    ├── ...
  │   │   │    └── custom_name_2_30.jpg
  │   │   ├── ...
  │   │   │    ├── ...
  │   │   │    ├── ...
  │   │   │    ├── ...
  │   │   │    └── ...
  │   │   ├── id_360000/
  │   │   │    ├── custom_name_360000_26.jpg
  │   │   │    ├── custom_name_360000_27.jpg
  │   │   │    ├── ...
  └   └   └    └── custom_name_360000_30.jpg
```
  #### Notes: 
  1. If some ids have less than 30 images, you can create new images using augmentation techniques.
  2. Choose the train, test and val images completely random.
  
  ### Feature Extraction
  For speed up, we use features instead of images. So first extract features of all images in dataset using a strong pretrained model. </br>
  Of course it is better that you train a model by scratch.
  For extract features, we used this [onnx pretrained model](https://github.com/onnx/models/blob/main/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx). ([Reference](https://github.com/onnx/models/tree/main/vision/body_analysis/arcface)) </br>
  By default, we saved extracted features [here](./features/), but you can change the path in [config file](./config/config.json). </br>
  The file structure for features is:
  ```
  ├── glint360k/
  │   ├── train/
  │   │   ├── id_1.npz
  │   │   ├── id_2.npz
  │   │   ├── ...
  │   │   └── id_360000.npz
  │   ├── val/
  │   │   ├── id_1.npz
  │   │   ├── id_2.npz
  │   │   ├── ...
  │   │   └── id_360000.npz
  │   ├── test/
  │   │   ├── id_1.npz
  │   │   ├── id_2.npz
  │   │   ├── ...
  └   └   └── id_360000.npz
```

  ### Model
  Now, you are ready for create the model and train it.

  ## Save Readme ✨  
  Once you're done, click on the save button to download and save your ReadMe!
  