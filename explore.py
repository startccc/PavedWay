#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from tqdm import tqdm
from aggregate_patches2 import aggregate_sale
import PIL
from PIL import Image
import pygsp as pg


# In[2]:


# SAVE_FOLDER = './GT/'
SAVE_FOLDER = './groundtruth/'
FOLDER = "./data/groundtruth/"


# # 1.- Functions & utilities

# In[3]:


def super_pixellize(img, T = 0.3, old_image_size_lat=1280, old_image_size_lon=1280, super_pixel_size=40):
    new_image_size_lat = old_image_size_lat // super_pixel_size
    new_image_size_lon = old_image_size_lon // super_pixel_size
    out_img = np.zeros(shape=(new_image_size_lat, new_image_size_lon))
    print('new_image_size: ', new_image_size_lat, 'x', new_image_size_lat) 
    for i in range(out_img.shape[0]):  # iterate through rows
        for j in range(out_img.shape[1]):  # iterate through columns
            i_old = i*super_pixel_size
            j_old = j*super_pixel_size
            patch_tmp = img[i_old:i_old+super_pixel_size, j_old:j_old+super_pixel_size]
            values = (np.mean(patch_tmp) > T)
            out_img[i, j]  = values
    return out_img


# In[4]:


# images
def convert_input(img):
    image = img[:,:,0]
    image[np.where( image!= 0)] = 1
    return image 
    
def convert_inputs(imgs_array):
    """
    :param imgs_array: array of images
    :return: list of filtered images"""
    out_images = []
    for image in imgs_array:
        out_images.append(convert_input(image))
    return out_images


# ## 1.1.- post-process groundtruth

# In[4]:


directory_in_str = './masks_machine/'
directory = os.fsencode(directory_in_str)

images = []
for file in tqdm(os.listdir(directory)):
    filename = os.fsdecode(file)
    img=mpimg.imread('./masks_machine/'+filename)
    image = convert_input(img)
    images.append(image)
    plt.imsave(SAVE_FOLDER+filename, image)


# ## 1.2.- Blurring stuff for JB

# In[ ]:


def pour_JB(one_is_transparent = True, use_numpy=False):
    filename = './JB_1.jpg'
    if use_numpy is True:
        img = mpimg.imread(filename)
        B = np.array(img)
        f4_dim = np.ones(B[:,:,0].shape)
        
        if one_is_transparent:
            f4_dim[np.where(B[:,:,0]!=0)] = 0
        else:
            f4_dim[np.where(B[:,:,0]!=0)] = 0
        f4_dim = f4_dim.reshape(880, 1280,1)
        B2 = np.concatenate((B, f4_dim), axis=2)
        # B2[np.where(B2[:,:,:]!=0)].shape
        mpimg.imsave('JBB_inv.png', B2)
    else:
        rgb_img = plt.imread(filename)
        blurred_image = rgb_img.filter(ImageFilter.BLUR)
        


# In[4]:


from PIL import ImageFilter


# In[16]:


filename = './JB_1.jpg'
rgb_img = plt.imread(filename)
img = PIL.Image.fromarray(np.uint8(rgb_img))
img = np.array(img)
img[np.where(img!=0)] = 255
img = PIL.Image.fromarray(np.uint8(rgb_img))
blurred_image = img.filter(ImageFilter.GaussianBlur(radius=2))
blurred_image.save('JBBB.png')
# blurred_image


# # 2.- Gross image:

# In[270]:


img_gross.shape

A = img_gross


# In[271]:


A[np.where(A!=0)] = 1


# In[272]:


A.shape


# In[273]:


indices.shape


# In[274]:


indices = indices[:, indices[0,:].argsort()]


# # 3.- Pygsp

# ## 3.1 Chauderon

# In[5]:


# list of images in chauderon
images_list = ['image_46.523185_6.624418.png',
    'image_46.523338_6.624481.png', 
    'image_46.523506_6.624492.png',
    'image_46.523617_6.624644.png',
    'image_46.523694_6.624738.png',
    'image_46.523795_6.624746.png',
    'image_46.523845_6.624975.png',
    'image_46.523850_6.624649.png',
    'image_46.523877_6.624858.png',
    'image_46.523940_6.624502.png',
    'image_46.524076_6.624297.png',
    'image_46.524197_6.624120.png',
    'image_46.524301_6.623933.png',
    'image_46.524402_6.623783.png',
    'image_46.524521_6.623620.png',
    'image_46.524629_6.623417.png',
    'image_46.524738_6.623269.png',
    'image_46.524863_6.623040.png',
    'image_46.524968_6.622873.png',
    'image_46.525087_6.622691.png',
    'image_46.525224_6.622474.png',
    'image_46.525360_6.622448.png',
    'image_46.525518_6.622223.png',
    'image_46.525608_6.622086.png']


# In[7]:


Chauderon_aggregated = aggregate_sale(FOLDER = FOLDER, images_list = images_list)


# In[8]:


# Rescale the data
Chauderon_aggregated[np.where(Chauderon_aggregated!=215)] = 0
Chauderon_aggregated[np.where(Chauderon_aggregated==215)] = 1
plt.imshow(Chauderon_aggregated)


# In[20]:


# superpixellize!
Chauderon_coarse = super_pixellize(Chauderon_aggregated, T = 0.3, old_image_size_lat=Chauderon_aggregated.shape[0], 
                            old_image_size_lon=Chauderon_aggregated.shape[1], super_pixel_size=100)
fig = plt.figure()
plt.imshow(Chauderon_coarse)
plt.axis('off')
fig.savefig('coarse_chauderon.png', transparent=True)


# In[10]:


indices = np.array(np.where(Chauderon_coarse == 1))


# #### 3.1.2.- Graph

# In[11]:


graph_indices =  np.fliplr(indices.T.copy())
print(graph_indices.shape)
graph_indices[:,1] = -graph_indices[:,1]
graph_indices[:,0] = graph_indices[:,0]
G = pg.graphs.NNGraph(graph_indices, k=10,order=2, epsilon=100)


# In[12]:


G.is_connected()


# In[13]:


fig = plt.figure()
pg.plotting.plot(G,show_edges=True) # , style=','
plt.grid(linestyle='--')
# plt.axis('off');


# In[14]:


graph_indices[:,0].shape


# In[15]:


right    = np.argmax(graph_indices[:,0])
left = np.argmin(graph_indices[:,0])
bottom   = np.argmin(graph_indices[:,1])
top  = np.argmax(graph_indices[:,1])

extremities = list(set([top, bottom, left, right]))


# In[18]:


# plot the signal
signal = np.random.rand(graph_indices.shape[0])
signal[extremities]=1;
fig, ax = plt.subplots(figsize=(16,16));
pg.plotting.plot_signal(G, signal=signal, vertex_size=160, show_edges=True,
                       backend='matplotlib', highlight=extremities, ax=ax,
                       colorbar=False, plot_name='')
plt.axis('equal');
fig.savefig('chauderon.png', transparent=True)


# ### 3.2 EPFL 

# In[11]:


images_list = ['image_46.517897_6.566332.png',
               'image_46.518105_6.566256.png',
               'image_46.518348_6.566113.png',
               'image_46.518467_6.566049.png',
               'image_46.520217_6.565059.png',
               'image_46.520362_6.565044.png']


# In[15]:


EPFL_aggregate = aggregate_sale(FOLDER = FOLDER, images_list = images_list)


# In[16]:


# clean the image
EPFL_aggregate[np.where(EPFL_aggregate!=215)] = 0
EPFL_aggregate[np.where(EPFL_aggregate==215)] = 1


# In[17]:


plt.imshow(EPFL_aggregate)


# In[19]:


EPFL_coarse = super_pixellize(EPFL_aggregate, T = 0.3, old_image_size_lat=EPFL_aggregate.shape[0],
                              old_image_size_lon=EPFL_aggregate.shape[1], super_pixel_size=100)


# In[20]:


fig = plt.figure()
plt.imshow(EPFL_coarse)
fig.savefig('EPFL_coarse.png')


# In[21]:


# extract indices:
indices = np.array(np.where(EPFL_coarse == 1))
indices = indices[:, indices[0,:].argsort()]

# Create the graph (pygsp nesessary)
graph_indices =  np.fliplr(indices.T.copy())
print(graph_indices.shape)
graph_indices[:,1] = -graph_indices[:,1]
graph_indices[:,0] = graph_indices[:,0]
G = pg.graphs.NNGraph(graph_indices, k=10,order=2, epsilon=100)


# In[22]:


# Get the extremities indices
right    = np.argmax(graph_indices[:,0])
left = np.argmin(graph_indices[:,0])
bottom   = np.argmin(graph_indices[:,1])
top  = np.argmax(graph_indices[:,1])

extremities = list(set([top, bottom, left, right]))


# In[26]:


# plot the signal
signal = np.random.rand(graph_indices.shape[0])
signal[extremities]=1;
fig, ax = plt.subplots(figsize=(16,16));
pg.plotting.plot_signal(G, signal=signal, vertex_size=80, show_edges=True,
                       backend='matplotlib', highlight=extremities, ax=ax,
                       colorbar=False, plot_name='')
# plt.axis('off');
plt.axis('equal');
fig.savefig('epfl.png', transparent=True)


# In[ ]:




