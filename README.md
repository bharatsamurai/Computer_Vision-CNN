# Computer_Vision-CNN
Computer Vision using CNN

------------Introduction------------------------------------------------------------------

Algorithm used : Convolution Nueral Network -- "Inception v3"

------------Steps to install and run the algorithm-----------------------------------------

1. Install Docker (installing Tensorflow using Docker)
2. Download Train Data (Folder_name: task). Folder 'task' contains subfolders 'cats' and 'dogs'
3. Start Docker with local files available (linking Tensorflow Image)
	$ sudo docker run -it -v $HOME/tf_files:/tf_files  gcr.io/tensorflow/tensorflow:latest-devel 

4. Retrieve the training code
	$ cd /tensorflow
	$ git pull

5. Retrain the Inception v3 model
	$ python tensorflow/examples/image_retraining/retrain.py \
	  --bottleneck_dir=/tf_files/bottlenecks \
	  --how_many_training_steps 500 \
	  --model_dir=/tf_files/inception \
	  --output_graph=/tf_files/retrained_graph.pb \
	  --output_labels=/tf_files/retrained_labels.txt \
	  --image_dir /tf_files/task

6. Inside Docker run the labelling script
	$ python /tf_files/label_image.py /tf_files/test/

7. Classifciation: A new csv file is generated "/tf_files/sample_submission.csv" having all the 'test' images with labels (1: Dog, 0: cat)

------------Only Testing Steps-----------------------------------------

1. Start Docker with local files available (linking Tensorflow Image)
	$ sudo docker run -it -v $HOME/tf_files:/tf_files  gcr.io/tensorflow/tensorflow:latest-devel 

2. Put all the images to be tested and labelled in the folder 'test'

3. Inside Docker run the labelling script
	$ python /tf_files/label_image.py /tf_files/test/

4. Classification: A new csv file is generated "/tf_files/sample_submission.csv" having all the 'test' images with labels (1: Dog, 0: cat)
5. Sort the "/tf_files/sample_submission.csv" by column 'id'
