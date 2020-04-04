import numpy as np
import cv2
import queue 
from keras.applications.nasnet import preprocess_input

def get_windows(padded_image, windows_x, windows_y, w_size, height, width) :

    # This function performs a sliding window through the padded image and returns a list of all windows
    X = np.ndarray((windows_x*windows_y, 96, 96, 3))
    for x in range(windows_x):
        for y in range(windows_y):
        #x is start of window, x+3 is end (extra mirror window is counted), because window is 3*window size. same for y 
            
            start_x = min(x * w_size, height-96)
            end_x = min((x+3) * w_size, height)
            start_y = min(y * w_size, width-96)
            end_y = min((y+3) * w_size, width)
            cur_im = padded_image[start_x:end_x, start_y:end_y]
            X[x*windows_y + y, :, :, :] = preprocess_input(cur_im)

    return X        

def get_predictions(model, image, windows_x, windows_y, w_size, height, width) :
    
    # This function performs the prediction on the list of images and returns the prediction matrix
    # which is a matrix where each element corresponds to the prediction of a window of the padded image

    # create padded image so prediction can be done on the whole original image with the 
    # sliding window method
    padded_image = np.pad(image, pad_width=((w_size, w_size), (w_size, w_size), (0, 0)), mode='symmetric')

    # Creating array of windows of the image
    X = get_windows(padded_image, windows_x, windows_y, w_size, height, width)

    # Performing the predictions
    Y = ((model.predict(X).ravel()*model.predict(X[:, ::-1, :, :]).ravel()*model.predict(X[:, ::-1, ::-1, :]).ravel()*model.predict(X[:, :, ::-1, :]).ravel())**0.25).tolist()
    
    # Putting the predictions in useful matrix form    
    preds = np.zeros((windows_x, windows_y))
    for x in range(windows_x):
        for y in range(windows_y):
            preds[x, y] = Y[x*windows_y + y]


    return preds

# This code implements the Connected Components Labelling algorithm using the classic method
# which is used to get the labeled regions of the image windows for use in the next part
# The input is the matrix of predictions, the output is labeled component matrix 

def CCL(preds) :

    windows_x = preds.shape[0]
    windows_y = preds.shape[1]  
    
    labels = np.zeros((windows_x, windows_y))
    cur_g = 1
    parent = {} # Dict to store disjoint sets
    
    # first pass
    for x in range(windows_x):
        for y in range(windows_y):
            if preds[x, y] > 0.5 :
                
                # Initializing the labels of the neighboring pixels (the one before and the one above)
                prev_x, prev_y = 0, 0                 
                if x>0 and labels[x-1, y] :
                    prev_x = labels[x-1, y]                    
                if y>0 and labels[x, y-1] :
                    prev_y = labels[x, y-1]   
                      
                # If both of the neighboring pixels aren't labeled    
                if not(prev_x or prev_y) :
                    labels[x, y] = cur_g
                    cur_g+=1
                    
                # If at least one of them is labeled    
                else :
                    # If only one of them is labeled, assign the value of the other one
                    if not prev_x:
                        parent[(x, y-1)] = (x, y)
                        labels[x, y] = prev_y
                    elif not prev_y :
                        parent[(x-1, y)] = (x, y)
                        labels[x, y] = prev_x
                    # If both of them are labeled, assign the minimal value and make the maximum child of the minimum     
                    else :
                        if prev_x >= prev_y :
                            parent[(x-1, y)] = (x, y-1)
                            parent[(x, y-1)] = (x, y)
                            labels[x, y] = prev_y
                        elif prev_y > prev_x :    
                            parent[(x, y-1)] = (x-1, y)
                            parent[(x-1, y)] = (x, y)
                            labels[x, y] = prev_x
                          
                            
    # second pass

    for x in range(windows_x):
        for y in range(windows_y):
            if preds[x, y] > 0.5 and (x, y) in parent : 
                px = (x, y)                
                q = queue.LifoQueue() # essentially a stack

                # Backtrack to the root, saving the pixels in the process
                while px in parent and parent[px] != px :
                    q.put(px)
                    px = parent[px]
                rlabel = labels[px[0], px[1]]  
                
                # Going back, labelling them in the way
                while not q.empty() :
                    px = q.get()
                    labels[px[0], px[1]] = rlabel
                    
    return labels 
                    
# This function will return the labeled boxes from the CCL
def get_labeled_boxes(labels):

    windows_h = labels.shape[0]
    windows_v = labels.shape[1]
    groups = {}

    # We run the program through the labels matrix a last time, getting the min and max x,
    # min and max y to draw bounding box
    for x in range(windows_h):
        for y in range(windows_v):
            if labels[x, y] :
                label = labels[x, y]
                if label in groups:
                    groups[label][0] = min(groups[label][0], x)
                    groups[label][1] = min(groups[label][1], y)
                    groups[label][2] = max(groups[label][2], x)
                    groups[label][3] = max(groups[label][3], y)
                else :
                    groups[label] = [x, y, x, y]

    return groups


# This function will return a copy of the image where the bounding boxes are highlights 
# each wtih a distinct color
def get_labeled_image(image, groups, w_size, height, width, alpha) :

    # We first initialize the colors for each group at random 
    """
    colors = np.random.uniform(0, 255, size=(len(groups.keys()), 3))
    color_dic = {}
    for i, k in enumerate(groups.keys()) :
        color_dic[k] = colors[i]
    """
    # Edit : Blue would be better

    # Then we create the labeled image with bouding boxes
    labeled_img = image.copy()

    for g, box in groups.items() :
        start_x = min(box[0] * w_size, height - 32)
        end_x = min((box[2]+1) * w_size, height)
        start_y = min(box[1] * w_size, width - 32)
        end_y = min((box[3]+1) * w_size, width)
        #in cv2 axes are the inversion of what I've used earlier and vice versa 
        cv2.rectangle(labeled_img, (start_y, start_x), (end_y, end_x), (255, 0, 0), 2)      

    return labeled_img           
                
def get_filled_image(image, groups, windows_x, windows_y, w_size, labels, height, width, alpha) :

    # We create a copy and an overlay (where the regions are highlighted) then 
    # merge them and return the resulting image
    filled_img = image.copy()
    overlay = image.copy()

    for x in range(windows_x):
        for y in range(windows_y):
            if labels[x, y]:
                start_x = min(x * w_size, height - 32)
                end_x = min((x+1) * w_size, height)
                start_y = min(y * w_size, width - 32)
                end_y = min((y+1) * w_size, width)
                #in cv2 axes are the inversion of what I've used earlier and vice versa 
                cv2.rectangle(overlay, (start_y, start_x), (end_y, end_x), (0, 0, 255), cv2.FILLED)
                    
    cv2.addWeighted(overlay, alpha, filled_img, 1 - alpha, 0, filled_img)

    return filled_img             
                
                
                
                
                
                
                
                
                