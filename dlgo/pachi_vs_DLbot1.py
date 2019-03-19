from dlgo.gtp.play_local import LocalGtpBot
from dlgo.agent.termination import PassWhenOpponentPasses
from dlgo.agent.predict import load_prediction_agent
import h5py

bot = load_prediction_agent(h5py.File("checkpointsOneLarge/checkpointsOneLarge/large_model_epoch_8.h5", "r"))
gtp_bot = LocalGtpBot(go_bot=bot, termination=PassWhenOpponentPasses(), handicap=0, opponent='pachi')
gtp_bot.run()
