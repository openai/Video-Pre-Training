from argparse import ArgumentParser
import pickle
import cv2
import numpy as np

from agent import ENV_KWARGS
from inverse_dynamics_model import IDMAgent


MESSAGE = """
This script will take a video, predict actions for its frames and
and show them with a cv2 window.

Press any button the window to proceed to the next frame.
"""

def main(model, weights, video_path, n_batches, n_frames):
    print(MESSAGE)
    agent_parameters = pickle.load(open(model, "rb"))
    net_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = IDMAgent(idm_net_kwargs=net_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights)

    required_resolution = ENV_KWARGS["resolution"]
    cap = cv2.VideoCapture(video_path)
    for _ in range(n_batches):
        print("=== Loading up frames ===")
        frames = []
        for _ in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break
            assert frame.shape[0] == required_resolution[1] and frame.shape[1] == required_resolution[0], "Video must be of resolution {}".format(required_resolution)
            # BGR -> RGB
            frames.append(frame[..., ::-1])
        frames = np.stack(frames)
        print("=== Predicting actions ===")
        predicted_actions = agent.predict_actions(frames)

        for i in range(n_frames):
            frame = frames[i]
            for y, (action_name, action_array) in enumerate(predicted_actions.items()):
                current_prediction = action_array[0, i]
                cv2.putText(
                    frame,
                    f"{action_name}: {current_prediction}",
                    (10, 10 + y * 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 255, 255),
                    1
                )
            # RGB -> BGR again...
            cv2.imshow("MineRL IDM model predictions", frame[..., ::-1])
            cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = ArgumentParser("Run IDM on MineRL recordings.")

    parser.add_argument("--weights", type=str, required=True, help="Path to the '.weights' file to be loaded.")
    parser.add_argument("--model", type=str, required=True, help="Path to the '.model' file to be loaded.")
    parser.add_argument("--video-path", type=str, required=True, help="Path to a .mp4 file (Minecraft recording).")
    parser.add_argument("--n-frames", type=int, default=128, help="Number of frames to process at a time.")
    parser.add_argument("--n-batches", type=int, default=10, help="Number of batches (n-frames) to process for visualization.")

    args = parser.parse_args()

    main(args.model, args.weights, args.video_path, args.n_batches, args.n_frames)
