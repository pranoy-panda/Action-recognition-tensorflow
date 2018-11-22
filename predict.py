#prediction for the 3D CNN
import tensorflow as tf
import numpy as np
import os,glob,cv2

img_rows,img_cols,img_depth=120,120,15
X_test = []
## loading the test videos
test_video_path = "test_videos/"
video_files = os.listdir(test_video_path)
## preprocessing the input(same as we had done during the training)
for video_file  in video_files:
	vid = test_video_path+video_file
	frames = []
	cap = cv2.VideoCapture(vid)
	for k in xrange(img_depth):
	    ret, frame = cap.read()
	    frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
	    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	    gray = np.reshape(gray,(120,120,1))
	    frames.append(gray)
	cap.release()
	input=np.array(frames)
	## reshaping the input for the model
	input = np.reshape(input,(1,15,120,120,1))
	X_test.append([input,video_file])
	#print video_file,input.shape	

tf.reset_default_graph()
## Let us restore the saved model 
sess = tf.Session()
# Step-1: Recreate the network graph. At this step only graph is created.
saver = tf.train.import_meta_graph('act_recog_3D_CNN_model.ckpt.meta')
# Step-2: Now let's load the weights saved using the restore method.
saver.restore(sess, tf.train.latest_checkpoint('./'))

# Accessing the default graph which we have restored
graph = tf.get_default_graph()

# Now, let's get hold of the op that we can be processed to get the output.
# In the original network y_pred is the tensor that is the prediction of the network
y_pred = graph.get_tensor_by_name("y_pred:0")

keep_prob = graph.get_tensor_by_name("keep_prob:0")

## Let's feed the images to the input placeholders
x= graph.get_tensor_by_name("x:0") 
y_true = graph.get_tensor_by_name("y_true:0") 
y_test_images = np.zeros((1, len(os.listdir('kth_dataset')))) 
y_test_images = y_test_images.astype(int)
for item in X_test:
	x_batch = item[0]
	### Creating the feed_dict that is required to be fed to calculate y_pred 
	feed_dict_testing = {x: x_batch, y_true: y_test_images,keep_prob: 1.0}
	result=sess.run(y_pred, feed_dict=feed_dict_testing)
	# result is of this format [probabiliy_of_rose probability_of_sunflower]
	print(np.argmax(result),item[1])
	print(result,item[1])
