#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/get_pet_labels.py
#                                                                             
# PROGRAMMER: 
# DATE CREATED:                                  
# REVISED DATE: 
# PURPOSE: Create the function get_pet_labels that creates the pet labels from 
#          the image's filename. This function inputs: 
#           - The Image Folder as image_dir within get_pet_labels function and 
#             as in_arg.dir for the function call within the main function. 
#          This function creates and returns the results dictionary as results_dic
#          within get_pet_labels function and as results within main. 
#          The results_dic dictionary has a 'key' that's the image filename and
#          a 'value' that's a list. This list will contain the following item
#          at index 0 : pet image label (string).
#
##
# Imports python modules
from os import listdir

# TODO 2: Define get_pet_labels function below please be certain to replace None
#       in the return statement with results_dic dictionary that you create 
#       with this function
# 

def get_pet_labels(image_dir: str):
    """
    Creates a dictionary of pet labels (results_dic) based upon the filenames 
    of the image files. These pet image labels are used to check the accuracy 
    of the labels that are returned by the classifier function, since the 
    filenames of the images contain the true identity of the pet in the image.
    Be sure to format the pet labels so that they are in all lower case letters
    and with leading and trailing whitespace characters stripped from them.
    (ex. filename = 'Boston_terrier_02259.jpg' Pet label = 'boston terrier')
    Parameters:
    image_dir - The (full) path to the folder of images that are to be
                classified by the classifier function (string)
    Returns:
      results_dic - Dictionary with 'key' as image filename and 'value' as a 
      List. The list contains for following item:
        index 0 = pet image label (string)
    """
    file_names: list = listdir(image_dir)
    results_dic = dict()

    for i, _ in enumerate(file_names):
      pet_label = get_pet_list(file_names)
      if file_names[i][0] != '.':
        if file_names[i] not in results_dic:
          results_dic[file_names[i]] = pet_label[i]
        else:
          print(f"** Warning! Key {file_names[i]} already exist with value {results_dic[pet_label[i]]}")

    return results_dic

pet_label: list = []

def get_pet_list(file_names: list):
    """
      Create a list of pet label (pet names) lists from th filenames (listdir)
      of the image file

      Parameters:
        file_names: function argument is of the listdir('pet_images'). 
                    i.e file_names = listdir('pet_images')
        Returns:
          Lists:     index 0 = [pet image label], pet_label[] a list of pet names
    """
    pet_label: list = []
    for names in file_names:
      pet_names = names.lower().split('.')[0]
      p_names = ''.join(alpha for alpha in pet_names if alpha.isalpha())
      pet_label.append([p_names])
    return pet_label
