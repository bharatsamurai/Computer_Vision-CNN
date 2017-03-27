import tensorflow as tf, sys, os, glob, csv

# change this as you see fit
image_path = sys.argv[1]

#creating the csv file fopr labelling of test images
c = csv.writer(open("tf_files/sample_submission.csv", "wb"))
c.writerow(["id", "label"])

with tf.gfile.FastGFile("tf_files/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

for file in glob.iglob('%s/*' %image_path):
    print(file)

    image_data = tf.gfile.FastGFile(file, 'rb').read()
    label_lines = [line.rstrip() for line 
                       in tf.gfile.GFile("tf_files/retrained_labels.txt")]

    #with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
        predictions = sess.run(softmax_tensor, \
                {'DecodeJpeg/contents:0': image_data})
    
        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            
            #writing in the csv file th labels for testing images
            if human_string == "dogs" and score > 0.70000:
                c.writerow([os.path.basename(file).split('.')[0], 1])
            if human_string == "cats" and score > 0.70000:
                c.writerow([os.path.basename(file).split('.')[0], 0])
            
    image_data = None
