import os  
import tensorflow as tf  
import keras.backend.tensorflow_backend as KTF  
  
def get_session(gpu_fraction=0.3):  
    '''''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''          
    num_threads = os.environ.get('OMP_NUM_THREADS')
    print ("num_threads={}".format(num_threads))
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)  
    print (gpu_options)
                  
    if num_threads:  
        return tf.Session(config=tf.ConfigProto(  
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))  
    else:  
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))  

