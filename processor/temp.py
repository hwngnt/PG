# SuperFastPython.com
# example of using the queue between processes without blocking
from time import sleep
from random import random
from multiprocessing import Process
from multiprocessing import Queue
from queue import Empty
import cv2
# generate work
def producer(queue):
    count = 0
    video = cv2.VideoCapture(0)
    print('Producer: Running', flush=True)
    while video.isOpened():
            flag, frame = video.read()
            queue.put(frame)
            count += 1
            print("[1] Push success", queue.qsize())
            if count == 90:
                break
    print('Producer: Done', flush=True)# consume work
def consumer(queue):
    print('Consumer: Running', flush=True)
    # consume work
    while True:
        # get a unit of work
        try:
            item = queue.get(block=False)
            print(item.shape)
        except Empty:
            print('Consumer: got nothing, waiting a while...', flush=True)
        # check for stop
        if item is None:
            break
        # report
        print(f'>got {item}', flush=True)
    # all done
    print('Consumer: Done', flush=True)# entry point
if __name__ == '_main_':
    # create the shared queue
    queue = Queue()
    # start the consumer process
    consumer_process = Process(target=consumer, args=(queue,))
    consumer_process.start()
    # start the producer process
    producer_process = Process(target=producer, args=(queue,))
    producer_process.start()
    # wait for all processes to finish
    producer_process.join()
    consumer_process.join()