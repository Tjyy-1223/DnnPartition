balance = 0
import threading
from threading import Lock
lock = Lock()

def change_it(n):
    # 先存后取，结果应该为0:
    nonlocal balance
    balance = balance + n
    balance = balance - n

def run_thread(n):
    for i in range(10):
        lock.acquire()
        try:
            change_it(n)
        finally:
            lock.release()

t1 = threading.Thread(target=run_thread, args=(5,))
t2 = threading.Thread(target=run_thread, args=(8,))
t1.start()
t2.start()
t1.join()
t2.join()
print(balance)