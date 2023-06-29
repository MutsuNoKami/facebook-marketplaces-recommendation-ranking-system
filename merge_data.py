import pandas as pd
import numpy as np
from clean_tabular_data import get_data_pipeline, clean_time
import os
from PIL import Image
import pickle
import seaborn as sns 
import matplotlib.pyplot as plt


def merge():

    products_data = get_data_pipeline()
    images_data = pd.read_csv("images.csv")
    images_data.drop(columns='Unnamed: 0', inplace=True)

    images_data['create_time'] = clean_time(images_data['create_time'])

    df = pd.merge(left=products_data, right=images_data, on=('product_id', 'create_time'), sort=True)
    df.drop(columns=['product_id','create_time', 'bucket_link', 'image_ref'], inplace=True)
    df.rename(columns={'id': 'image_id'}, inplace=True)

    df.category = df.category.apply(lambda x: x.split('/')[0]) 
    df.category =  df.category.astype('category')
    df['category_codes'] =  df.category.cat.codes 
    decoder_dict = dict(enumerate(df['category'].cat.categories))

    df.to_csv('training_data.csv') 