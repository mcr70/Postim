import matplotlib.pyplot as plt

def show_image_data_sample(train_generator):
    plt.figure(figsize=(6, 6))
    (X_batch, Y_batch) = train_generator.next()

    for i in range(9):
        plt.subplot(3, 3, (i + 1))
        plt.imshow(X_batch[i])

    train_generator.reset()  # as used once, can have strange orders otherwise
    plt.show()

def draw_stats(history_callback):
    plt.plot( history_callback.history["loss"], color='blue', label='loss')
    plt.plot( history_callback.history["accuracy"], color='red', label='accuracy')

    plt.legend(loc='best')
    plt.show()
