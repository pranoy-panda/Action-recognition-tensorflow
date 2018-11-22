import numpy as np
import cv2
import os

k = 0
def get_kth_data(batch_size):

    # image specification
    img_rows,img_cols,img_depth=120,120,15
    num_chan = 1

    # Training data

    # variable to store entire dataset
    X_tr=[]           

    #Reading boxing action class

    video_folder_path = os.listdir('kth_dataset/')

    for video in video_folder_path:
        listing = os.listdir('kth_dataset/'+video+'/')
        print video
        for vid in listing:
            vid = 'kth_dataset/'+video+'/'+vid
            frames = []
            cap = cv2.VideoCapture(vid)
            fps = cap.get(5)
            #print "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)
          
            for k in xrange(img_depth):
                ret, frame = cap.read()
                frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = np.reshape(gray,(120,120,1))
                frames.append(gray)
            cap.release()

            input=np.array(frames)

            #print input.shape
            #ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
            #print ipt.shape

            X_tr.append(input)

    X_tr_array = np.array(X_tr)   # convert the frames read into array

    num_samples = len(X_tr_array) 
    print num_samples

    #Assign Label to each class

    label=np.ones((num_samples,6),dtype = int)
    '''
    boxing
    handclapping
    handwaving
    running
    jogging
    walking
    '''
    label[0:100]   = np.array([1,0,0,0,0,0])
    label[100:199] = np.array([0,1,0,0,0,0])
    label[199:299] = np.array([0,0,1,0,0,0])
    label[299:399] = np.array([0,0,0,1,0,0])
    label[399:499] = np.array([0,0,0,0,1,0])
    label[499:]    = np.array([0,0,0,0,0,1])


    train_data = [X_tr_array,label]

    (X_train, y_train) = (train_data[0],train_data[1])
    print 'X_Train shape:', X_tr_array.shape
	
    
    #np.reshape(label,(train_set.shape[0],1))
    #print train_set[k:k+batch_size,:,:,:].shape
    #print label[k:k+batch_size,:].shape
    return X_tr_array,label
    

def get_kth_data_next_batch(X_tr_array,batch_size,label):
    global k
    #k = np.random.randint(0,X_tr_array.shape[0]-batch_size)
    var = X_tr_array[k:k+batch_size,:,:,:],label[k:k+batch_size,:] 
    k+=batch_size
    if k>=X_tr_array.shape[0]:
        k=0
    return var
