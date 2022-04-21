import argparse
import yaml
from agent import Agent
# import config


# def main(mode):
    





if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train', dest = 'train', action= 'store_true', help='train model')
    parser.add_argument('--eval', dest='eval', action='store_true', help ='eval model')
    parser.add_argument('--env', dest='env', default ='PongNoFrameskip-v4', type = str, help = 'gym environment')
    parser.add_argument('--config', dest='config', default='config.yaml')
    args = parser.parse_args()
    # * store_true의 경우 default 값은 false이며, 인자를 적어 주면 true가 저장된다

    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    config['env_id'] = args.env

    if args.train:
        agent = Agent(config)
        agent.train()

    elif args.eval:
        agent = Agent(config)
        agent.eval(2)
