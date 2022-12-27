# AdvisIL - A Class-Incremental Learning Advisor

In this repository, we share the code for reproducing the results of our article "AdvisIL - A Class-Incremental Learning Advisor" accepted at WACV 2023. This repository also aims at facilitating contributions to the set of reference experiments used by AdvisIL's recommender system. 

**Abstract**

_Recent class-incremental learning methods combine deep neural architectures and learning algorithms to handle streaming data under memory and computational constraints. The performance of existing methods varies depending on the characteristics of the incremental process. To date, there is no other approach than to test all pairs of learning algorithms and neural architectures on the training data available at the start of the learning process to select a suited algorithm-architecture combination. 
To tackle this problem, in this article, we introduce AdvisIL, a method which takes as input the main characteristics of the incremental process (memory budget for the deep model, initial number of classes, size of incremental steps) and recommends an adapted pair of learning algorithm and neural architecture. The recommendation is based on a similarity between the user-provided settings and a large set of pre-computed experiments.
AdvisIL makes class-incremental learning easier, since users do not need to run cumbersome experiments to design their system.  
We evaluate our method on four datasets under six incremental settings and three deep model sizes. We compare six algorithms and three deep neural architectures. Results show that AdvisIL has better overall performance than any of the individual combinations of a learning algorithm and a neural architecture._

<img
  src="captions/advisIL_principle.png"
  alt="Alt text"
  title="AdvisIL's principle"
  style="display: inline-block; margin: 0 auto; max-width: 250px">

**How to cite**

Feillet Eva, Petit Gégoire, Popescu Adrian, Reyboz Marina, Hudelot Céline, "AdvisIL - A Class-Incremental Learning Advisor", Proceedings of the Winter Conference on Applications of Computer Vision. 2023. 
_____

## Tutorial 

__Outline__

1. Check the requirements
2. Get the datasets
3. Explore neural architectures
4. Tune hyperparameters
5. Take a look at some preliminary scaling experiments
6. Get familiar with a few incremental learning algorithms
7. Compute reference experiments for AdvisIL using reference scenarios
8. Compute recommendations using AdvisIL 
9. Wrapping up 


**Content of this repository** 

Subfolders 
* captions  
* config_utils : automatically generate config files for the experiments 
* FeTrIL : incremental learning algorithm  
* hp_tuning : sample scripts for hyperparameter tuning 
* imagenet : how to build custom imagenet subsets 
* images_list_files : utility functions and image lists for loading the data 
* LUCIR : incremental learning algorithm 
* models : neural network architectures and utility functions  
* reco : AdvisIL recommendations
* results : pre-computed experimental results 
* scaling : scaling experimetns and utility functions 
* SIW : incremental learning algorithm 
* SPBM : incremental learning algorithm 
* py37_requirements.txt : requirements file to reproduce the environment
* py37_requirements_linux64.txt : idem (Linux specific)


### 1. Check the requirements

We use Python deep learning framework PyTorch (`Python version 3.7`, `torch version 1.7.1+cu110`) in association with cuda (`CUDA Version: 11.4`).

In a future release, we plan to integrate our code with Avalanche continual learning library. Note that the packages used in this repository are compatible with the current version of Avalanche and ray[tune].

To start with, we recommend creating a virtual environment dedicated to this project using conda. Use the requirements file `requirements_py37.txt`, or alternatively the explicit `py37_requirements_linux64.txt` for a Linux distribution.

> conda create --name py37 --file requirements_py37.txt

To use this environment : 

> conda activate py37


### 2. Get the datasets

In our article we consider nine datasets. Six of them are sampled from ImageNet-21k. The others are Food101, Google Landmarks (v2) and iNaturalist (v2018).

__a) ImageNet subsets__

We used six datasets sampled from ImageNet-21k database. Three of them are thematic, and were
obtained by sampling leaf classes from ImageNet belonging to the “food”, “fauna”, and “flora” sub-hierarchies, respectively.
The three other datasets were obtained by randomly sampling classes from ImageNet. 
Each dataset contains 100 classes, with 340 images per class for training, and 60 images for testing. 
Each sampled class is only used in one dataset.

**Prerequisite** : access to [ImageNet-21K full database](https://www.image-net.org/).

For each ImageNet subset, run the following commands to (1) get the images in a dedicated folder, and for sanity check (2) compute the mean and standard deviation of the colour channels.  

* RANDOM0, RANDOM1 and RANDOM2 subsets

> python ./imagenet/build_datasets/imagenet_random_subsetter.py ./imagenet/build_datasets/imagenet_random_subsetter<0|1|2>.cf

> python ./images_list_files/compute_dataset_mean_from_images_list.py ./images_list_files/train100/imagenet_random_<0|1|2>/train.lst 

* FLORA, FAUNA and FOOD subsets

> python ./imagenet/build_datasets/imagenet_subsetter.py ./imagenet/build_datasets/imagenet_subsetter_<fauna|flora|food>.cf

> python ./images_list_files/compute_dataset_mean_from_images_list.py ./images_list_files/train100/imagenet_<fauna|flora|food>/train.lst 


NB : __Sanity check__ . You must obtain the same mean and standard deviation values as in [`datasets_mean_std.txt`](./images_list_files/datasets_mean_std.txt) . 

*Optional* : To create your own ImageNet subset, see the dedicated [tutorial](./imagenet/tutorial.md).

 __b) Other datasets__

Please download the following datasets : 

* [Food101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
* [Google Landmarks v2](https://github.com/cvdfoundation/google-landmark)
* [iNaturalist 2018](https://github.com/visipedia/inat_comp/tree/master/2018)

Note that in the case of iNaturalist, there is no need to download the data related to the semantic segmentation task of the original competition. Also, we provide the labels in our image lists (see subfolder [`images_list_files`](./images_list_files/)).

__c) Note on reproductibility__ 

To facilitate reproductibility, we provide in the [`images_list_files`](./images_list_files/) folder the _explicit list of images_ contained in each of the datasets we used. These lists are fed to the dataloaders when training or testing models.

For example, [images_list_files/food101/train.lst](./images_list_files/food101/train.lst) contains the list of training images for the Food101 dataset. 

An image list file is structured as follows : 
* The first line of the text file should contain the root path to your local version of the dataset, and "-1" is used as flag for this root path. _NB : Please change the root path to your own._
> <path_to_the_dataset>/food101/images -1
* All other lines are in the format "<class_name>/<image_name> <class_number>". See example below. 
> apple_pie/1005649.jpg 87

Note that some lists may contain more data than we actually used : e.g. in our experiments we only consider the first 100 classes of Food101, leaving out the last one. But we provide a complete split for the 101 classes in case you wish to use them all. 


### 3. Explore neural architectures

We have made preliminary experiments to study the impact of architecture on the incremental performance of small convolutional neural networks. Therefore we have implemented versions of ResNet, MobileNetv2 and ShuffleNetv2 which allow to scale each architecture according to its width (number of convolutional filters) and depth (number of building blocks). These custom architectures are implemented [here](./models/).

To explore the architectures, run the following script. You will see the number of parameters corresponding to various scaling operations on 3 network types (ResNet18 with BasicBlocks, Mobilenetv2 and ShuffleNetv2).

> python ./models/scaler.py


### 4. Tune hyperparameters 

Before running the main experiments, we searched for suitable hyperparameters for each neural architecture. 

_Prerequisite_ : ray-tune. The following command installs Ray and dependencies for Ray Tune.

> pip install -U "ray[tune]"  

To test your configuration, run this minimum working example with just 2 epochs.

> python ./hp_tuning/hp_tuner.py ./hp_tuning/hp_tuner_test.cf

You can now perform a hyperparameter search for each backbone. We provide an example configuration file for each backbone.

> python ./hp_tuning/hp_tuner.py ./hp_tuning/hp_tuner_<mobilenet|resnet|shufflenet>.cf

Modify the paths to run more hyperparameter tuning experiments using other datasets. 

The next step consists in collecting and visualizing results.  

> python ./hp_tuning/tuning_analyser.py


### 5. Take a look at some preliminary scaling experiments

We have run scaling experiments to propose a scaling heuristic in the case of a class-incremental learning task. 
In the following scripts, we provide examples of scaling experiments using the LUCIR algorithm and the datasets iNaturalist and imagenet_random_0. 
Similar steps and observations apply for other methods and datasets (see below for more details on each incremental learning algorithm).

1. Define settings and get yaml files

> python ./scaling/scaling_yaml_writer.py

2. Build config files, folder structure and launcher files

First, set paths and other parameters in `config_writer.cf`, in particular the path to your input yaml file and the path to your output config and launcher files.

> python ./LUCIR/codes/config_writer.py ./LUCIR/configs/config_writer.cf 

3. Run scaling experiments using the launcher files

_Recommendation_ : As this study involves numerous experiments, we recommend to split the experiments across several launcher files (manually), depending on the available computing resources and on the number of trials you wish to run for each particular experimental setting. Make sure to give the jobs adapted names and to name the log and error files in such a way that they don't conflict with each other and can be found easily. 

4. Parse log files and plot results

_Reminder : Adapt the name of the output files._

For reference, we provide log files with accuracies and visualisations [here](./scaling/logs/). 

Running the following command, parse the log files. The output is a `.csv` file containing the average incremental accuracy of each experiment.

> python ./scaling/log_parser.py ./scaling/configs/log_parser_<dataset>.cf

Finally, plot the average incremental accuracy for each architecture (one plot per backbone). 

> python ./scaling/WACV_scaling_plots.py


### 6. Get familiar with a few incremental learning algorithms 

In our article, we report experiments with six recent class-incremental learning algorithms : LUCIR \[1\], SPB-M\[2\], DeepSLDA\[3\], SIW\[4\], DeeSIL\[5\] and FeTrIL\[6\]

\[1\]Hou, Saihui, et al. "Learning a unified classifier incrementally via rebalancing." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019. 

\[2\]Wu, Guile, Shaogang Gong, and Pan Li. "Striking a balance between stability and plasticity for class-incremental learning." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.

\[3\]Hayes, Tyler L., and Christopher Kanan. "Lifelong machine learning with deep streaming linear discriminant analysis." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops. 2020.

\[4\]Belouadah, Eden, Adrian Popescu, and Ioannis Kanellos. "Initial classifier weights replay for memoryless class incremental learning."  British Machine Vision Conference (BMVC). 2020.

\[5\]Belouadah, Eden, and Adrian Popescu. "DeeSIL: Deep-Shallow Incremental Learning." Proceedings of the European Conference on Computer Vision (ECCV) Workshops. 2018. 

\[6\]Petit, Grégoire, et al. "FeTrIL: Feature Translation for Exemplar-Free Class-Incremental Learning." Proceedings of the Winter Conference on Applications of Computer Vision. 2023. 


In this repository, we share implementations of these algorithms that, when needed, we adapted, so that they are able to handle : 
- custom datasets (e.g. not just ImageNet-1k and CIFAR100)
- custom neural architectures (e.g. not just ResNet18 and ResNet32)
- custom incremental learning scenarios (e.g. not only scenarios with an equal repartition of classes across all incremental states, but also scenarios with more classes in the initial state). 
- examplar-free scenarios : we do not use any memory buffer in our experiments.


#### a. LUCIR 

Our implementation is based on this [original repository](https://github.com/hshustc/CVPR19_Incremental_Learning). LUCIR has initially been proposed as a
learning algorithm with memory of past examples. In practice, as we focus on examplar-free class-incremental learning, we set the size of LUCIR’s memory buffer to zero.

Example code for launching LUCIR.

> python ./LUCIR/codes/main.py ./LUCIR/configs/LUCIR.cf

_Useful tip: For convenience, we reuse the first state obtained by running LUCIR for experiments which use a fixed feature extractor, namely DeepSLDA, SIW, DeeSIL and FeTrIL._ 

#### b. SPB-M

TODO modif learning rate cf Slack !!

We use the SPB-M version, which uses a data augmentation procedure based on image rotations. As no official code with released for this method, we based our implementation on LUCIR’s implementation, with a modified loss function and SPB-M’s data augmentation procedure. Unfortunately, our implementation does not reac the same accuracy as in the original paper. Contributions are welcome to improve this. 

Example code for launching SPB-M.

> python ./SPBM/codes/main.py ./SPBM/configs/SPBM.cf

#### c. DeepSLDA 

Our implementation is based on the [original repository](https://github.com/tyler-hayes/Deep_SLDA) of Tyler Hayes.

#### d. SIW attention j'ai renommé le dossier siw -> SIW

Our implementation is based on the [original repository](https://github.com/EdenBelouadah/class-incremental-learning/tree/master/siw) of Eden Belouadah.

i - First batch
> python ./SIW/FT/codes/scratch.py ./SIW/FT/configs/scratch.cf

ii - Fine-tuning without memory
> python ./SIW/FT/codes/no_mem_ft.py ./SIW/FT/configs/no_mem_ft.cf

iii - Feature extraction
> python ./SIW/FT/codes/features_extraction.py ./SIW/FT/configs/features_extraction.cf

iv - Weight correction using standardization of initial weights
> python ./SIW/FT/codes/inFT_siw.py ./SIW/FT/configs/inFT_siw.cf

#### e. DeeSIL 

Our implementation is based on this [original repository](https://github.com/EdenBelouadah/class-incremental-learning/tree/master/deesil).

See our [dedicated repository](https://github.com/GregoirePetit/DeeSIL).

#### f. FeTrIL

Our implementation of FeTrIL is shared [here](https://github.com/GregoirePetit/FeTrIL).

NB : you might find references to "MobIL" in the code, it is actually the same as FeTrIL.

### 7. Compute reference experiments for AdvisIL using reference scenarios

1. Generate yaml files with your hyperparameters for each backbone type

NB : hyperparameters for the first batch were obtained by tuning. Hyperparameters for incremental states are more generic. 

Don't forget to modify the source and destination folders in the config files. You may store all yaml files in the same folder since a filtering step is applied on yaml filename (what algorithm name do they contain) prior to config file generation (next step).

Reference points

> python ./config_writer/yaml_writer.py

Test points

> python ./config_writer/yaml_writer_testsets.py

For debugging purposes, use this version that creates quicker experiments : 

> python ./config_writer/yaml_writer_test.py

2. Generate folders, config files and launcher files for each method

> python ./config_writer/config_writer_lucir.py ./config_writer/config_writer_lucir.cf

> python ./config_writer/config_writer_spbm.py ./config_writer/config_writer_spbm.cf

> python ./config_writer/config_writer_siw.py ./config_writer/config_writer_siw.cf

Do the same thing for test experiments. 

3. Split or group experiments across launchers (manually) and run them

A typical arborescence is the following : TO ADD (screenshot ?)

4. Parse log files and save results in a structured format


### 8. Compute recommendations using AdvisIL

Parsed results for reference configurations and for test configurations are provided in csv format [here](./results/). 

Compute the recommendations and reproduce the results of the paper by running the following command.

> python ./reco/advisil_reco_vote.py ./results/ref_configs.csv ./results/test_configs.csv 


### 9. Wrapping up

In this repository, we provide a detailed tutorial to :
- reproduce the results presented in our WACV 2023 article "AdvisIL - A Class-Incremental Learning Advisor",
- contribute to AdvisIL's recommendations by adding your own results to the database of pre-computed experiments.

AdvisIL facilitates the choice of a suited pair of CIL algorithm and backbone network for a user-defined incremental learning scenario. 
AdvisIL requires little information from the user and provides a recommendation by leveraging trends observed on pre-computed experiments from a set of reference configurations. 
Our evaluation indicates that AdvisIL is effective, as it often provides a relevant recommendation.

As AdvisIL is thought as a collaborative tool, don't hesitate to contribute to this repository by :
- reporting issues
- adding algorithms, backbones and datasets
- adding experimental results.
Thanks for your contribution !
