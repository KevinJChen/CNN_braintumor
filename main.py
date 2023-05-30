import tensorflow as tf
from tensorflow.keras import layers
import h5py
import glob
import numpy as np
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score

def main():

    # 3064 images split into 4 folders of 766 images
    dataset_paths = ['data/brainTumorDataPublic_1766',
                     'data/brainTumorDataPublic_7671532',
                     'data/brainTumorDataPublic_15332298',
                     'data/brainTumorDataPublic_22993064']

    '''
    cjdata.label: 1 for meningioma, 2 for glioma, 3 for pituitary tumor
    cjdata.PID: patient ID
    cjdata.image: image data
    cjdata.tumorBorder: a vector storing the coordinates of discrete points on tumor border
    cjdata.tumorMask: a binary image with 1s indicating tumor region
    '''

    # np array of all images in the dataset
    # should be 3064 images in total
    all_images = np.empty((3064,), dtype=object)
    all_images_index = 0

    # iterate across folders
    for data_path in dataset_paths:
        mat_files = glob.glob(data_path + "/*.mat")

        # iterate across files in folder
        for mat_file in mat_files:
            image = load_mat_file_image(mat_file)
            label = load_mat_file_label(mat_file)

            # resize the image to 256 (for normalization)
            resized_image = resize_to_256(image)
            all_images[all_images_index] = (resized_image, label)
            all_images_index += 1

            if len(resized_image) != 256:
                print(len(resized_image))

    # first approach:
    # randomly divide into 10 approximately equal portions

    # shuffle the array
    np.random.shuffle(all_images)

    # split the array into 10 random sub arrays
    random_splits = np.array_split(all_images, 10)

    # primary test data
    for i in range(0, 10):
        # initialize data
        x_train = np.empty((3064-len(random_splits[i]),), dtype=object)
        y_train = np.empty((3064-len(random_splits[i]),))

        x_test = np.empty((len(random_splits[i]),), dtype=object)
        y_test = np.empty((len(random_splits[i]),))

        # indexes
        train_index = 0
        test_index = 0

        # add elements for x_test and y_test
        for image in random_splits[i]:
            x_test[test_index] = image[0]
            y_test[test_index] = image[1]
            test_index += 1
            if test_index > 307:
                print('bigger')

        # add elements for x_train and y_train
        for j in range(0, 10):
            if i == j:
                continue

            for image in random_splits[j]:
                x_train[train_index] = image[0]
                y_train[train_index] = image[1]
                train_index += 1

    # preprocess the data
    nested_tensors = []
    for arr in x_train:
        nested_tensors.append(tf.convert_to_tensor(arr))
    x_train_tensor = tf.stack(nested_tensors)

    nested_tensors = []
    for arr in x_test:
        nested_tensors.append(tf.convert_to_tensor(arr))
    x_test_tensor = tf.stack(nested_tensors)

    y_train_tensor = tf.convert_to_tensor(y_train)
    y_test_tensor = tf.convert_to_tensor(y_test)

    # train the model using the training data
    model = define_model()
    model.fit(x_train_tensor, y_train_tensor, batch_size=32, epochs=10, validation_data=(x_test_tensor, y_test_tensor))

    # evaluate the model's performance
    test_loss, test_accuracy = model.evaluate(x_test_tensor, y_test_tensor)
    print(test_loss, test_accuracy)

    # get the predictions from the model
    predictions = model.predict(x_test_tensor)

    # convert the predicted probability to class labels
    predicted_labels = np.argmax(predictions, axis=1)

    # calculate the precision
    precision = precision_score(y_test, predicted_labels, average='weighted')

    # calculate the recall
    recall = recall_score(y_test, predicted_labels, average='weighted')

    # calculate the f1 score
    f1 = f1_score(y_test, predicted_labels, average='weighted')

    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 Score:', f1)

# loads the .mat file given a file path
# returns the image data only
def load_mat_file_image(filepath):
    mat_data = h5py.File(filepath)
    image = mat_data['cjdata']['image']
    return image

def load_mat_file_label(filepath):
    mat_data = h5py.File(filepath)
    label = mat_data['cjdata']['label']
    return label[0][0]-1

# resize the image data to 256 if it is not already
def resize_to_256(image):

    # convert image to np array
    np_image = np.array(image)

    # convert np array to pil image
    image_pil = Image.fromarray(np_image)

    # resize the image
    resized_image_pil = image_pil.resize((256, 256))

    # convert resized image to np array
    resized_image_data = np.array(resized_image_pil)

    return resized_image_data


# CNN model
def define_model():

    # define the CNN model
    #model = tf.keras.Sequential([
        # 1 -> image input: 256x256 x 1 images
        # 2 -> convolutional: 16 5x5x1 convolutions with stride [2 2] and padding 'same'
        #tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(2, 2),
        #                       padding='same', input_shape=(256, 256, 1)),
        # 3 -> rectified linear unit: rectified linear unit
        #tf.keras.layers.ReLU(),
        # 4 -> drop out: 50% dropout
        #tf.keras.layers.Dropout(0.5)
        # 5 -> max pooling: 2x2 max pooling with stride [2 2] and padding [0 0 0 0]
        # 6 -> convolutional: 32 3x3x16 convolutions with stride [2 2] and padding 'same'
        # 7 -> rectified linear unit: rectified linear unit
        # 8 -> dropout: 50% dropout
        # 9 -> max pooling: 2x2 max pooling with [2 2] and padding [0 0 0 0]
        # 10 -> convolutional: 64 3x3x32 convolutions with stride [1 1] and padding 'same'
        # 11 -> rectified linear unit: rectified linear unit
        # 12 -> dropout: 50% dropout
        # 13 -> max pooling: 2x2 max pooling with stride [2 2] and padding [0 0 0 0]
        # 14 -> convolutional: 128 3x3x64 convolutions with stride [1 1] and padding 'same'
        # 15 -> rectified linear unit: rectified linear unit
        # 16 -> dropout: 50% dropout
        # 17 -> max pooling: 2x2 max pooling with stride [2 2] and padding [0 0 0 0]
        # 18 -> fully connected: 1025 hidden neurons in fully connected (FC) layer
        # 19 -> rectified linear unit: rectified linear unit
        # 20 -> fully connected: 3 hidden neurons in fully connected layer
        # 21 -> softmax: softmax
        # 22 -> classification output:
        # 3 output classes, "1" for meningioma, "2" for glioma, and "3" for a pituitary tumor


    #])

    model = tf.keras.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])

    # compile the CNN model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    #model.summary()

    return model


if __name__ == "__main__":
    main()

