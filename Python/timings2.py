import tensorflow as tf
# tf.compat.v1.disable_eager_execution()

@tf.function
def func(x,y):
    return tf.math.exp(x)*y+y[0,0]

if __name__=="__main__":
    import time
    x=tf.random.normal((10000,1),dtype=tf.float32)
    y=tf.random.normal((1,10000),dtype=tf.float32)
    tic=time.time()
    z=func(x,y)
    ctime=time.time()-tic
    print(f'Elapsed time {ctime} s with {z[0,0]}')
    tic=time.time()
    z=func(x,y)
    ctime=time.time()-tic
    print(f'Elapsed time {ctime} s with {z[0,0]}')
    tic=time.time()
    z=func(x,y)
    ctime=time.time()-tic
    print(f'Elapsed time {ctime} s with {z[0,0]}')