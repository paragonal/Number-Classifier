import numpy as np

def get_labels(filename):
    file = open(filename, "rb")

    labels = []

    magic_number= file.read(4)
    num_items = int.from_bytes(file.read(4),"big")
    for i in range(num_items):
        labels.append(int.from_bytes(file.read(1),"big"))
    return labels


def get_images(filename):
    file = open(filename, "rb")

    images = []

    magic_number= file.read(4)
    num_items = int.from_bytes(file.read(4),"big")
    num_rows = int.from_bytes(file.read(4), "big")
    num_cols = int.from_bytes(file.read(4), "big")

    for i in range(num_items):
        im = np.array(list(file.read(num_cols*num_rows)))
        im=im.reshape(num_rows,num_cols)
        images.append(im)
    return images