import argparse

from RL.train_ddqn import train as train_ddqn
from RL.train_ppo import train as train_ppo

def main():
    parser = argparse.ArgumentParser(description="UAV RL Training")

    parser.add_argument("--algo", required=True, choices=["ddqn", "ppo"])
    parser.add_argument("--config", default=None, help="Optional config override")

    args = parser.parse_args()

    if args.algo == "ddqn":
        print("Running DDQN training...")
        train_ddqn()
    elif args.algo == "ppo":
        print("Running PPO training...")
        train_ppo()

if __name__ == "__main__":
    main()
