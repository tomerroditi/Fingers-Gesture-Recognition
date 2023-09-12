import sys
sys.path.append('../')

import numpy as np
import Source.fgr.models as models

from Source.fgr.pipelines import Data_Pipeline
from Source.fgr.data_manager import Data_Manager
from warnings import simplefilter
from pathlib import Path
from importlib import reload
import matplotlib.pyplot as plt
from pathlib import Path 
from tqdm import tqdm

import numpy as np
from PIL import Image
from threading import Thread, Lock
import os

def create_image_from_array(array, filename):
    image = Image.fromarray(array, mode="L")
    image.save(filename)
    # print(f"Image {filename} created successfully!")

def process_image_batch(arrays, labels, base_directory, thread_id, num_threads, lock, progress_bar):
    num_images = len(arrays)
    # print(f"Thread {thread_id} processing {num_images} images")

    batch_size = (num_images + num_threads - 1) // num_threads
    start_index = thread_id * batch_size
    end_index = min((thread_id + 1) * batch_size, num_images)

    for i in range(start_index, end_index):
        array = arrays[i]
        # print(array.shape)
        label = labels[i].split('_')[3]
        directory = os.path.join(base_directory, str(label))
        os.makedirs(directory, exist_ok=True)
        filename = os.path.join(directory, f"image_{label}_{i}.png")

        create_image_from_array(array, filename)

        progress_bar.update(1)

def create_images_from_arrays(arrays, labels, base_directory, num_threads):
    threads = []
    lock = Lock()

    num_images = len(arrays)
    progress_bar = tqdm(total=num_images, desc="Creating Images")
    for thread_id in range(num_threads):
        thread = Thread(target=process_image_batch, args=(arrays, labels, base_directory, thread_id, num_threads, lock, progress_bar))
        thread.start()
        threads.append(thread)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    progress_bar.close()


def main():


    label_dict = {
        "TwoFingers":0,
        "ThreeFingers":1,
        "Abduction":2,
        "Fist":3,
        "Bet":4,
        "Gimel":5,
        "Het":6,
        "Tet":7,
        "Kaf":8,
        "Nun":9,
    }

    # pipeline definition and data manager creation
    data_path = Path('../../data/doi_10')
    pipeline = Data_Pipeline(base_data_files_path=data_path)  # configure the data pipeline you would like to use (check pipelines module for more info)
    subject = 1
    dm = Data_Manager([subject], pipeline)
    print(dm.data_info())


    for i in range(1,4):
        dataset = dm.get_dataset(experiments=[f'{subject:03d}_*_{i}'])
        data = dataset[0]
        labels = dataset[1]

        create_images_from_arrays(data.reshape(data.shape[0],4,4), labels, f'../../data/doi_10/emg16x1_matplotlib/{subject:03d}_{i}', 256)

if __name__ == "__main__":

    # array = np.random.rand(4,4)
    # create_image_from_array(array, "test.png")
    main()

