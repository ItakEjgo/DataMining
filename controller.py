import multiprocessing as mp
import random

from function import get_directory, search_increase


class ParallelWorker(mp.Process):
    def __init__(self, in_queue, out_queue, random_seed):
        super(ParallelWorker, self).__init__(target=self.start)
        self.inQ = in_queue
        self.outQ = out_queue
        random.seed(random_seed)

    def run(self):
        while True:
            task = self.inQ.get()
            path = task[0]
            sol = search_increase(path)
            self.outQ.put(sol)


def create_worker(num):
    workers = []
    for i in range(num):
        workers.append(ParallelWorker(mp.Queue(), mp.Queue(), random.randint(0, 10 ** 9)))
        workers[i].start()
    return workers


def finish_worker(workers):
    for w in workers:
        w.terminate()


if __name__ == '__main__':
    # /home/kaiqiang/file/2016data/SH000001.csv
    print("Please input the directory path")
    path = input()
    print("Please input the cpu number")
    cpu_num = int(input())
    fp = get_directory(path)
    workers = create_worker(cpu_num)
    l = len(fp)
    cou = 0
    while cou < l:
        result = [None for _ in range(cpu_num)]
        for i in range(cpu_num):
            if cou + i < l:
                workers[i].inQ.put((fp[cou + i],))

        for i in range(cpu_num):
            if cou + i < l:
                result[i] = workers[i].outQ.get()
        for i in range(cpu_num):
            print(f'{result[i][0]}:{result[i][1]}')
        cou += cpu_num
    finish_worker(workers)
