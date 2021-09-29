## Import des librairies et modules nécessaires au fonctionnement de l'application spark

import boto3
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.image import ImageSchema
from pyspark.sql.functions import lit
from pyspark.sql.functions import udf
from pyspark.sql.functions import explode
from pyspark.sql.types import *
from functools import reduce
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import PCA
import cv2 as cv

## 0. (Ré)Initialiser la session Spark

sc = SparkContext.getOrCreate()
spark = SparkSession.builder.appName("Prevot-P8").getOrCreate()

## 1. Importation des images dans un dataframe Spark

def image_df(bucket='p8-prevot', prefix = 'data/Test/'):
    
    '''Permet de créer un dataframe spark contenant les images et leurs caractéristiques'''
    client = boto3.client('s3')
    result = client.list_objects(Bucket=bucket, Prefix=prefix, Delimiter='/')

    dataframes = []
    
    for o in result.get('CommonPrefixes'):
        
        folder = o.get('Prefix')
        sub_folder = folder.split('/')[2]
        data_path = 's3://{}/{}'.format(bucket, folder)
        images_df = ImageSchema.readImages(data_path, recursive=True).withColumn("label", lit(sub_folder))
        dataframes.append(images_df)
        
    df = reduce(lambda first, second: first.union(second), dataframes)
    df = df.repartition(200)
    
    return df

df = image_df()
print('Création du dataframe contenant les images: OK')

## 2. Prétraitement: Detection et extraction de features avec descripteurs ORB

def orb_descriptors(img):
    
    height = img[1]
    width = img[2]
    nchannels = img[3]
    data = img[5]
    
    img_array = np.array(data).reshape(height, width, nchannels)
    
    orb = cv.ORB_create(nfeatures=30)
    kp, des = orb.detectAndCompute(img_array,None)
    
    des = des.tolist()
    
    return des

orb_UDF = udf(lambda img: orb_descriptors(img), ArrayType(ArrayType(IntegerType())))
df = df.withColumn("Descripteurs ORB", orb_UDF("image"))
print('ORB_UDF OK')


df = df.select(df['image'], df['label'], explode(df['Descripteurs ORB']))
df = df.withColumnRenamed("col","Descripteur")
print('Explode OK')

list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
df = df.withColumn("Descripteur-vect", list_to_vector_udf(df["Descripteur"]))
df = df.drop('Descripteur')
print('Vectorization OK')

## 3. Réduction de dimension avec une ACP

pca = PCA(k=20, inputCol="Descripteur-vect", outputCol="Descripteur-ACP")
pca = pca.fit(df)
df = pca.transform(df)
df = df.drop('Descripteur-vect')
print('PCA OK')

# 4. Enregistrer les résultats sur le répertoire results de s3

df.write.parquet(path="s3://p8-prevot/results/", mode="overwrite")
print('Write OK')
