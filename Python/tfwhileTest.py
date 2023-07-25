import tensorflow as tf
@tf.function
def sumSquare(n):
  i, result = tf.constant(0), tf.constant(0)
  while i < n: # AutoGraph converts while-loop to tf.while_loop().
    result += i 
    i += 1
  return result

@tf.function
def sumSquare2(n):
  i, result = tf.constant(0), tf.constant(0)
  c = lambda i, _: tf.less(i, n)
  b = lambda i, result: (i + 1, result + i * i)
  return tf.while_loop(c, b, [i, result],parallel_iterations=1)[0]


@tf.function
def parallelSumSquare(m):
   return tf.map_fn(sumSquare,1*tf.range(0,1000),parallel_iterations=m)
   

if __name__=="__main__":
    import time 
    tic=time.time()
    parallelSumSquare(10)
    print(f'Elapsed time {time.time()-tic}')

    tic=time.time()
    parallelSumSquare(1)
    print(f'1 Elapsed time {time.time()-tic}')

    tic=time.time()
    parallelSumSquare(2)
    print(f'2 Elapsed time {time.time()-tic}')

    tic=time.time()
    parallelSumSquare(4)
    print(f'4 Elapsed time {time.time()-tic}')

    tic=time.time()
    parallelSumSquare(8)
    print(f'8 Elapsed time {time.time()-tic}')

    tic=time.time()
    parallelSumSquare(12)
    print(f'12 Elapsed time {time.time()-tic}')

    tic=time.time()
    parallelSumSquare(20)
    print(f'20 Elapsed time {time.time()-tic}')