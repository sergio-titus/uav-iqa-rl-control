from RL.train_ddqn import train as train_ddqn
from RL.train_ppo import train as train_ppo

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ddqn", "ppo"], required=True)
    args = parser.parse_args()

    if args.algo == "ddqn":
        train_ddqn()
    else:
        train_ppo()

if __name__ == "__main__":
    main()
