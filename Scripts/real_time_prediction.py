import pickle

from Source.fgr.models import Real_Time_Predictor
from Source.fgr.pipelines import Data_Pipeline
from Source.streamer.data import Data
from Source.streamer.viz import Viz

# load a model
model_fp = r'C:\Users\AaronGerston\PycharmProjects\GR-RT\models\GR-RT_pos1_aaron3gest_s001_BT_3gestures_100.00%.pkl'
with open(model_fp, 'rb') as f:
    model = pickle.load(f)

# conect to a streamer
signal_check = False  # set to True to check the signal before running the real time predictor
host_name = "127.0.0.1"
port = 20001
timeout = 20
verbose = False
data_streamer = Data(host_name, port, verbose=verbose, timeout_secs=timeout)
data_streamer.start()

# todo: make the signal check a util function
if signal_check:
    timeout = None
    verbose = True
    data_collector = Data(host_name, port, timeout_secs=timeout, verbose=verbose)
    data_collector.start()
    plotter = Viz(data_collector, plot_imu=False, plot_ica=False, ylim_exg=(-350, 350), update_interval_ms=50,
                  max_timeout=20)
    plotter.start()
    exit(0)

# configure the data pipeline - in future we will make sure it is saved a model attribute to make sure the data is
# processed the same way it was during training
pipeline = Data_Pipeline()

# create a real time predictor
predictor = Real_Time_Predictor(model, pipeline, vote_over=10, max_timeout=20)

while data_streamer.is_connected:
    prediction, confidence = predictor.majority_vote_predict()
    print(f'{prediction}, confidence: {confidence}')




