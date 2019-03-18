from dlgo import httpfrontend
from dlgo import mcts

BOARD_SIZE = 5


def main():
    # bot = mcts.MCTSAgent(700, temperature=1.4)
    bot = {'mcts': mcts.MCTSAgent(800, temperature=0.7)}
    web_app = httpfrontend.get_web_app(bot)
    web_app.run()


if __name__ == '__main__':
    main()