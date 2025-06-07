import numpy as np
import multiprocessing as mp
from time import time
import datetime


np.random.RandomState(100)
#arr = np.random.randint(0, 10, size=[20000, 5])
#data = arr.tolist()
arr = np.random.randint(0, 10, size=[2000000, 5])
#arr = np.random.randint(0, 10, size=[2000, 5])
#data[:5]
arr[:5, :]

def count_within_range(rows, minimum=4, maximum=8):
    count = 0
    for row in rows:
        for n in row:
            if minimum <=n <= maximum:
                count += 1
    return count

if __name__=='__main__':
    start = datetime.datetime.now()
    result_seq = count_within_range(arr)
    end = datetime.datetime.now()
    print('result sequential: ',result_seq)
    print('runtime of sequential: {} s'.format((end - start).total_seconds()))
    cpu_count = mp.cpu_count()
    row_size = arr.shape[0]
    start = 0
    delta = row_size//cpu_count
    #数据分块
    arr_chunks = []
    for i in range(cpu_count-1):
        start = i*delta
        arr_chunk = arr[start:start+delta, :]
        arr_chunks.append(arr_chunk)
    arr_chunk = arr[start+delta:, :]
    arr_chunks.append(arr_chunk)
    pool = mp.Pool(cpu_count)
    start = datetime.datetime.now()
    result_parallel =pool.map(count_within_range, arr_chunks)
    end = datetime.datetime.now()
    print('result parallel: ',sum(result_parallel))
    print('runtime of parallel: {} s'.format((end - start).total_seconds()))
    #下面两行代码采用不同的方式实现多进程并行
    #result =pool.map(count_within_range, [arr[i, :] for i in range(arr.shape[0])])
    result_parallel_async =pool.map_async(count_within_range, arr_chunks)
    print('result parallel async: ',sum(result_parallel_async.get(timeout=100)))
    pool.close()
    pool.join()


