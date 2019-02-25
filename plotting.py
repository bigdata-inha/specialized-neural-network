import matplotlib.pyplot as plt

def show_train_per_epochs(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

    return


def show_train_per_time(history, callback):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    top_k_acc = history.history['val_top_k_categorical_accuracy']

    times = callback.times

    plt.plot(times, acc, 'bo', label='Training acc')
    plt.plot(times, val_acc, 'b', label='Validation acc')
    plt.plot(times, top_k_acc, 'bs', label='Top 5 Validation acc')
    plt.title('Training and validation accuracy')
    plt.xticks([times[0], times[-1]], visible=True, rotation='horizontal')
    plt.legend()
    plt.figure()

    plt.plot(times, loss, 'bo', label='Training loss')
    plt.plot(times, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xticks([times[0], times[-1]], visible=True, rotation='horizontal')
    plt.legend()
    plt.show()
    return
