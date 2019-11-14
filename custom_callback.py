from keras import callbacks
from keras import backend
import time

class time_config(callbacks.Callback):
    def __init__(self, patience=5, max_reduce_num=3, reduce_rate=0.2):
        super(callbacks.Callback, self).__init__()
        self.patience = patience
        self.wait = 0
        self.best_score = -1.
        self.best_top5_score = -1.
        self.reduce_rate = reduce_rate
        self.max_reduce_num = max_reduce_num
        self.reduce_num = 0

    def on_train_begin(self, logs={}):
        self.times = []
        self.start_time = time.time()

    def on_epoch_begin(self, epoch, logs={}):

        self.epoch_time_start = time.time()
    def on_epoch_end(self, epoch, logs={}):
        elapsed_time = time.time() - self.start_time
        self.times.append(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

        # dacaying learning rate
        # training stop condition: decay learning rate more than 3
        current_score = logs.get('val_acc')
        if current_score > self.best_score:
            self.best_score = current_score
            self.wait = 0
        else:
            self.wait += 1
            if self.reduce_num < self.max_reduce_num:
                if self.wait >= self.patience:
                    lr = backend.get_value(self.model.optimizer.lr) * self.reduce_rate
                    backend.set_value(self.model.optimizer.lr, lr)
                    print("Change lr rate to %f" % (lr))
                    self.wait = 0
                    self.reduce_num += 1
            else:
                print("Epoch %d: early stopping" % (epoch))
                self.model.stop_training = True
                
