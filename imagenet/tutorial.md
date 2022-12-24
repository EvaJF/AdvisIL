# Create your own ImageNet subset


1. __Download the Wordnet package.__

We used Wordnet 3.0 to build our subsets of ImageNet dataset. To obtain the package, run the following commands (example for Ubuntu).

Sanity update command to update your package repositories
> sudo apt-get update -y

Install command with -y flag to quickly install WordNet package with its dependencies. 
> sudo apt-get install -y wordnet 


2. __Sanity check using the "bear" concept__ 

First, get the subtree corresponding to "bear". 

> wn bear -o -treen > <path/to/your/wn_tree_bear.txt>

Then, check that you get the same as in the reference file [bear.tree](AdvisIL/imagenet/wordnet_tree_files/bear.tree). 


3. __Choose a concept__ 

Choose a concept and get the classes subsumed by this concept in WordNet's hierarchy.

> wn <your_concept> -o -treen > <path/to/your/wn_tree_concept.txt>

You obtain a tree structure in a text file.

NB : if you get error `Search too large. Narrow search and try again...`, you might need to choose a concept lower in the hierarchy. Otherwise, the issue can be fixed by manually setting a larger buffer space in wordnet source code.


4. __Filter classes according to their number of images__ 

Select a portion of the previously obtained classes according to their number of available images.

[prepare_food_dataset.py](./prepare_food_dataset.py) is an example script for preparing an imagenet food dataset which includes classes containing at least X images.

How to run the script - Example of a food dataset with at least 350 images per class.

> cd AdvisIL/imagenet

> python3 prepare_food_dataset.py ./wordnet_tree_files/food.tree ./imagenet_leaves.lst ./synsets_words_size_map.txt 350 ./imagenet_food_example/your_output.txt

* `wn_food_tree.txt` contains WordNet subhierarchy with "food" as root.
* `imagenet_leaves.lst` is the list of all WordNet leave concepts (the identifiers, not the actual names).
synsets_words_size_map.txt contains the mapping of words to synsets.
* 350 is the minimum number of images per class
* `food_over_350.txt` is your output file containing the identifiers of 1000 classes respecting the topic and the image number constraints. 

[prepare_random_dataset.py](./prepare_random_dataset.py) is a similar script for preparing an imagenet dataset with random leaf classes (not thematic) containing at least X images.

In the [class_lists folder](./class_lists/) you find the text files containing our lists of classes for 6 custom ImageNet subsets used in the article.


5. __Get the images__

In this final step, you create a dedicated folder for your custom dataset, like in the main tutorial.

For a random subet, use [imagenet_random_subsetter.py](AdvisIL/imagenet/build_datasets/imagenet_random_subsetter.py). For a thematic subset, use [imagenet_subsetter.py](AdvisIL/imagenet/build_datasets/imagenet_subsetter.py).