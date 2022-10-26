# Create your own ImageNet subset

1. __Download the Wordnet package.__

For example, on Ubuntu : first, run a sanity update command to update your package repositories, then run the install command with -y flag to quickly install WordNet package with its dependencies. We used Wordnet 3.0 to build our subsets of ImageNet dataset.

> sudo apt-get update -y

> sudo apt-get install -y wordnet 

2. __Sanity check using the "bear" concept__ 

First, get the description for the "bear" concept. Then, check that you get the same as below : 

> wn bear -o -treen > <path/to/your/wn_tree_bear.txt>

> Hyponyms of noun bear
1 of 2 senses of bear
Sense 1
{02131653} bear
=> {01322983} bear cub
=> {02132136} brown bear, bruin, Ursus arctos
=> {02132466} Syrian bear, Ursus arctos syriacus
=> {02132580} grizzly, grizzly bear, silvertip, silver-tip, Ursus horribilis, Ursus arctos horribilis
=> {02132788} Alaskan brown bear, Kodiak bear, Kodiak, Ursus middendorffi, Ursus arctos middendorffi
=> {02132320} bruin
=> {02133161} American black bear, black bear, Ursus americanus, Euarctos americanus
=> {02133400} cinnamon bear
=> {02133704} Asiatic black bear, black bear, Ursus thibetanus, Selenarctos thibetanus
=> {02134084} ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus
=> {02134418} sloth bear, Melursus ursinus, Ursus ursinus
etc.

3. __Choose a concept__ 

Choose a concept and get the classes subsumed by this concept in WordNet's hierarchy.

> wn bear -o -treen > <path/to/your/wn_tree_concept.txt>


NB : if you get error `Search too large. Narrow search and try again...` ???

4. __Get classes with enough images__ 

Select a portion of the previously obtained classes according to the number of available images for each class.

"""
Script for preparing an imagenet food dataset which includes classes containing at least X images.

How to run the script - Example of a food dataset containing 1000 classes with at least 350 images per class.

python3 prepare_food_dataset.py path/to/wn_food_tree.txt path/to/imagenet_leaves.lst path/to/synsets_words_size_map.txt 350 path/to/food_over_350.txt

wn_food_tree.txt contains WordNet subhierarchy with "food" as root.
imagenet_leaves.lst is the list of all WordNet leave concepts (the identifiers, not the actual names).
synsets_words_size_map.txt contains the mapping of words to synsets.
350 is the minimum number of images per class
food_over_350.txt is your output file containing the identifiers of 1000 classes respecting the topic and the image number constraints. 

Example : 

> cd AdvisIL/build_datasets/wordnet

> python3 prepare_food_dataset.py files/wn_food_tree.txt files/imagenet_leaves.lst files/synsets_words_size_map.txt 350 files/food_over_350.txt

"""

to complete.