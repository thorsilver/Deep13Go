import h5py

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint

from dlgo.agent.predict import DeepLearningAgent, load_prediction_agent
from dlgo.data.parallel_processor import GoDataProcessor
from dlgo.encoders.sevenplane import SevenPlaneEncoder
from dlgo.httpfrontend import get_web_app
from dlgo.networks import large


def main():
    go_board_rows, go_board_cols = 19, 19
    nb_classes = go_board_rows * go_board_cols
    num_games = 15000

    encoder = SevenPlaneEncoder((go_board_rows, go_board_cols))
    processor = GoDataProcessor(encoder=encoder.name())
    generator = processor.load_go_data('train', num_games, use_generator=True)
    test_generator = processor.load_go_data('test', num_games, use_generator=True)

    input_shape = (encoder.num_planes, go_board_rows, go_board_cols)
    model = Sequential()
    network_layers = large.layers(input_shape)
    for layer in network_layers:
        model.add(layer)
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    epochs = 20
    batch_size = 128
    model.fit_generator(generator=generator.generate(batch_size, nb_classes),
                            epochs=epochs,
                            steps_per_epoch=generator.get_num_samples() / batch_size,
                            validation_data=test_generator.generate(batch_size, nb_classes),
                            validation_steps=test_generator.get_num_samples() / batch_size,
                            callbacks=[ModelCheckpoint('checkpointsSeven/deep7_epoch_{epoch}.h5')])

    model.evaluate_generator(generator=test_generator.generate(batch_size, nb_classes),
                                steps=test_generator.get_num_samples() / batch_size)

    deep_learning_bot = DeepLearningAgent(model, encoder)
    deep_learning_bot.serialize("checkpointsSeven/deep7.h5")

    model_file = h5py.File("checkpointsSeven/deep7.h5", "r")
    bot_from_file = load_prediction_agent(model_file)

    web_app = get_web_app({'predict': bot_from_file})
    web_app.run()


if __name__ == '__main__':
    main()