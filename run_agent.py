from argparse import ArgumentParser
import pickle
import time

from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival

from agent import MineRLAgent, ENV_KWARGS

def main(model, weights):
    env = HumanSurvival(**ENV_KWARGS).make()
    print("---Loading model---")
    agent_parameters = pickle.load(open(model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights)

    print("---Launching MineRL enviroment (be patient)---")
    obs = env.reset()

    frame_cap = 1.0/20 # set to 20 FPS
    time_1 = time.perf_counter()
    unprocessed = 0
    while True:
        can_render = False
        time_2 = time.perf_counter()
        passed = time_2 - time_1
        #check how much time has passed since last frame/pass round and how long the unprocessed queue has been waiting.
        unprocessed += passed 
        time_1 = time_2

        while(unprocessed >= frame_cap):
            unprocessed -= frame_cap    
            can_render = True   

        if can_render:
            minerl_action = agent.get_action(obs)
            obs, reward, done, info = env.step(minerl_action)
            env.render()


if __name__ == "__main__":
    parser = ArgumentParser("Run pretrained models on MineRL environment")

    parser.add_argument("--weights", type=str, required=True, help="Path to the '.weights' file to be loaded.")
    parser.add_argument("--model", type=str, required=True, help="Path to the '.model' file to be loaded.")

    args = parser.parse_args()

    main(args.model, args.weights)
