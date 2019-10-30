# imports
import pync # notification library
# time libs
import time, datetime
import os, shutil
import pyautogui
from PIL import Image, ImageFile
# computer vision library, includes pandas as pd and numpy
from fastai.vision import *

# untility functions

# function to create a directory
def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir   

# function to resize screenshots to 244x244 whilst keeping the aspect ratio
def resize_image(file, final_size=244):
    im = PIL.Image.open(file)
    size = im.size
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])
    im = im.resize(new_image_size, PIL.Image.ANTIALIAS)
    new_im = PIL.Image.new("RGB", (final_size, final_size))
    new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
    new_im.save(file, 'PNG', quality=90)



# output dictionary
tn_dict = {}

# model output over recording period
time_res = []
warn_count = 0

# inference time
rec_freq = 10 # screenshot frequency - every 10 seconds
rec_mins = 1 # record for n mins
screens_per_min = 60/rec_freq

rec_time = int(screens_per_min*rec_mins) # recording time

# target image size
final_size = 244

# alert threshold
warn_thresh = 3

# negative classes
neg_classes = ['facebook', 'netflix', 'youtube']

# load learner and transformations

defaults.device = torch.device('cpu') # set learner to make inference with CPU
# transforms
tfms = get_transforms(do_flip=False, flip_vert=False, max_rotate=0, 
                      max_zoom=1.2, max_lighting=0.15, max_warp=0, 
                      p_affine=0, p_lighting=0.25)
# load trained model
learn = load_learner('data')

# takes float between 0 and 1, returns integer as percent value for user output
def get_pct_int(n, rec_time): return int(n/rec_time*100)


# prediction loop
for i in range(rec_time):
    
    # get time_stamp and change format to YYYY-MM-DD_hh-mm-ss e.g. 2018-05-23_12-05-21, [:-7] removes second precision
    time_str = datetime.datetime.now().isoformat().replace("T","_").replace(":","-")[:-7]
    # set filename 
    fname = '_{}_{}.png'.format(time_str, i)
    # set destination folder
    fdir = 'screens/'
    # file
    file = fdir+fname
    # take screenshot
    pyautogui.screenshot(file)
    # resize image to match original model inputs - 244 x 244px with 3 color channels
    resize_image(file, final_size)
    
    img = open_image(file)
    
    # get predictions
    pred_class, pred_idx, output = learn.predict(img)
    # convert prediction to string
    pred_class = str(pred_class)
    # save file with prediction in name for future training
    os.rename(file, pred_class + fname)
    # turn output tensort into list
    preds = output.tolist()
    
    # threshold
    thresh = 0.2
    # create list of predictions with high prediction confidence    
    hc_preds = [ 'pred class: {}, pred value {}'.format(learn.data.classes[i], p) for i, p in enumerate(preds) if p > thresh]
    
    time_res.append(pred_class)
    
    # increase counter for warning notification
    if str(pred_class) in neg_classes:
        warn_count += 1
    
    # notify user if model has returned 'distracted' activity class more than 3 
    if warn_count >= warn_thresh and str(pred_class) in neg_classes:
        pync.notify(title='warning', message='focus!', sound='default')
        
    # print output to terminal - for debugging
    print('high confidence preds: {}, {}/{}'.format(hc_preds, i+1, rec_time))
    
    # pause for duration of recording frequency
    time.sleep(rec_freq) 
    
# list comprehension to count model results for user feedback
acts = [[act , get_pct_int(time_res.count(act), len(time_res))] for act in set(time_res)]

# create title string for user output
title_str = '{} min breakdown:'.format(rec_mins)
# reset output string
output_str = ''
# create output string
for i in acts:
    output_str += '{}: {}% '.format(i[0],i[1])

# notify user
pync.notify(title=title_str, message=output_str)