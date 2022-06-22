from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival

from agent import MineRLAgent, ENV_KWARGS


def main():
    env = HumanSurvival(**ENV_KWARGS).make()
    print("---Loading model---")
    agent = MineRLAgent(env)
    agent.load_weights("model.weights")

    print("---Launching MineRL enviroment (be patient)---")
    obs = env.reset()

    while True:
        minerl_action = agent.get_action(obs)
        obs, reward, done, info = env.step(minerl_action)
        env.render()


if __name__ == "__main__":
    main()
