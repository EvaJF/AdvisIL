#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys,os

"""
Example script for preparing an imagenet food dataset which includes 1000 classes.
python ./imagenet/prepare_food_dataset.py ./wordnet_tree_files/food.tree ./imagenet/imagenet_leaves.lst ./imagenet/synsets_words_size_map.txt 350 ./imagenet_food_example/food_over_350.txt
"""

wn_food_path = sys.argv[1] #path to the full wordnet food subset
wn_leafs_path = sys.argv[2] #path to the list of imagenet leaf synsets
wn_syns_path = sys.argv[3] #path to the full list of imagenet synsets + their associated sizes
min_size = int(sys.argv[4]) #min number of images for a synset to be retained
food_syns = sys.argv[5] #output file with the names of food synsets which have enough images associated to them

syn_sizes = {} #dictionary with synset names and number of images
f_full = open(wn_syns_path)
for fline in f_full:
	fparts = fline.rstrip().split("\t")
	syn_sizes[fparts[0]] = int(fparts[-1])
	#print(fparts[0],fparts[-1])
f_full.close()

#dictionary for leaves and their number of images, if this number is sufficient
leaves_sizes = {}
f_leaves = open(wn_leafs_path)
for lline in f_leaves:
	lline = lline.rstrip()
	#print(lline+"<")
	if lline in syn_sizes and syn_sizes[lline] >= min_size:
		leaves_sizes[lline] = syn_sizes[lline]
f_leaves.close()

print("OK leaves:",len(leaves_sizes))

#get the leaves which appear under "food" in wordnet and have a sufficient number of images associated to them
f_out = open(food_syns,"w")
food_cnt = 0
unique_syns = {}
f_food = open(wn_food_path)
for dline in f_food:
	if '}' in dline:
		dline = dline.rstrip().replace('{','}').split('}')
		crt_offset = "n"+dline[1]
		if crt_offset in leaves_sizes and not crt_offset in unique_syns:
			food_cnt = food_cnt+1
			f_out.write(crt_offset+"\t"+str(leaves_sizes[crt_offset])+"\n")
			unique_syns[crt_offset] = ""
			print(food_cnt,crt_offset,leaves_sizes[crt_offset])
f_food.close()
f_out.close()

