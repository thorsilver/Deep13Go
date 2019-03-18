import h5py

from dlgo.agent.predict import DeepLearningAgent, load_prediction_agent
from dlgo.httpfrontend import get_web_app

model_file = h5py.File("checkpointsOneLarge\large_model_epoch_8.h5", "r")
bot_from_file = load_prediction_agent(model_file)

web_app = get_web_app({'predict': bot_from_file})
web_app.run()