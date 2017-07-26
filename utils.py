import cv2
import numpy as np
import matplotlib.pyplot as plt


def preprocess_image(image):
    '''
    Method for preprocessing images
    '''
    # original shape: 160x320x3
    # apply subtle blur
    image = cv2.GaussianBlur(image, (3,3), 0)
    # convert to YUV color space (as nVidia paper suggests)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    return image


def augment_data(images, angles):
    # data augmentation by flipping
    augmented_images = images
    augmented_angles = angles
    for image, angle in zip(images, angles):
        image_flipped = np.fliplr(image)
        angle_flipped = -angle
        augmented_images.append(image_flipped)
        augmented_angles.append(angle_flipped)
    return augmented_images, augmented_angles


# visualization
def display_results(history_object):
    # plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
