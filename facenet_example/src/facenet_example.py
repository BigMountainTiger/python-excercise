'''
facenet model structure: https://github.com/serengil/tensorflow-101/blob/master/model/facenet_model.json
pre-trained weights https://drive.google.com/file/d/1971Xk5RwedbudGgTIrGAL4F7Aifu7id1/view?usp=sharing
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json
 
model = model_from_json(open("./model/facenet_model.json", "r").read())
model.load_weights('./model/facenet_weights.h5')
 
model.summary()

def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    
    return euclidean_distance

def findEuclideanSimilarity(source_representation, test_representation):
    threshold = 0.35
    
    source_representation = l2_normalize(source_representation)
    test_representation = l2_normalize(test_representation)
    
    euclidean_distance = findEuclideanDistance(source_representation, test_representation)
    
    return euclidean_distance < threshold, euclidean_distance, 

 
def findCosineSimilarity(source_representation, test_representation):
    threshold = 0.07
    
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    
    cosine_similarity = 1 - (a / (np.sqrt(b) * np.sqrt(c)))

    
    return cosine_similarity < threshold, cosine_similarity

def load_raw_image(path):
    img = image.load_img(path, target_size=(160, 160))
    
    return img
    
def load_image(path):
    raw_img = load_raw_image(path)
    img = image.img_to_array(raw_img)
    img = np.expand_dims(img, axis = 0)
    
    return raw_img, img

raw_img1, img1 = load_image("./processed-images/yuan.jpeg")
raw_img2, img2 = load_image("./processed-images/yuan-1.jpeg")

p1 = model.predict(img1)[0,:]
p2 = model.predict(img2)[0,:]

verified, euclidean_distance = findEuclideanSimilarity(p1, p2)

title1 = "人工智能认定是同一个罪犯" if verified else "人工智能认定不是同一个罪犯"
title2 = "\n欧几里德距离 - " + str(euclidean_distance)

font_path = "./model/simhei.ttf"
prop = font_manager.FontProperties(fname=font_path)

fig = plt.figure()
fig.set_size_inches(6, 4)
fig.suptitle(title1 + title2, fontproperties=prop)
plt.subplot(1, 2, 1)
plt.imshow(raw_img1)
plt.subplot(1, 2, 2)
plt.imshow(raw_img2)
plt.show()

