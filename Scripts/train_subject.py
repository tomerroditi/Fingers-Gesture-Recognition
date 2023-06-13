import os.path
import multiprocessing

from pathlib import Path
from Source.fgr import models
from Source.utils import data_collection, train_model


if __name__ == "__main__":
    # data collector params
    host_name = "127.0.0.1"  # IP address from which to receive data
    port = 20001  # Local port through which to access host
    data_dir = str(Path(os.path.dirname(os.path.abspath(__file__))).parent / 'data')

    # model initialization

    # Dvir's try #
    # Get the current directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Traverse up the directory structure twice to reach 'FGR' directory
    project_dir = os.path.dirname(script_dir)
    # Construct the absolute path to the image file
    img_dir = os.path.join(project_dir, 'images')
    n_classes = len(os.listdir(img_dir))
    # it worked!

    # n_classes = len(os.listdir('images'))
    model = models.Net(num_classes=n_classes, dropout_rate=0.1)

    manager = multiprocessing.Manager()
    lock = multiprocessing.Lock()
    shared_dict = manager.dict()

    p1 = multiprocessing.Process(target=data_collection, args=(host_name, port, data_dir, shared_dict, lock))
    p2 = multiprocessing.Process(target=train_model, args=(model, shared_dict, lock))

    p1.start()
    p2.start()

    p1.join()
    p2.join()
