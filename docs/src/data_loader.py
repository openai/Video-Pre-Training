# Code for loading OpenAI MineRL VPT datasets
# NOTE: This is NOT original code used for the VPT experiments!
#       (But contains all [or at least most] steps done in the original data loading)

import json
import glob
import os
import random
from multiprocessing import Process, Queue, Event

import numpy as np
import cv2

from run_inverse_dynamics_model import json_action_to_env_action
from agent import resize_image, AGENT_RESOLUTION

QUEUE_TIMEOUT = 10

CURSOR_FILE = os.path.join(os.path.dirname(__file__), "cursors", "mouse_cursor_white_16x16.png")

MINEREC_ORIGINAL_HEIGHT_PX = 720

# If GUI is open, mouse dx/dy need also be adjusted with these scalers.
# If data version is not present, assume it is 1.
MINEREC_VERSION_SPECIFIC_SCALERS = {
    "5.7": 0.5,
    "5.8": 0.5,
    "6.7": 2.0,
    "6.8": 2.0,
    "6.9": 2.0,
}


def composite_images_with_alpha(image1, image2, alpha, x, y):
    """
    Draw image2 over image1 at location x,y, using alpha as the opacity for image2.

    Modifies image1 in-place
    """
    ch = max(0, min(image1.shape[0] - y, image2.shape[0]))
    cw = max(0, min(image1.shape[1] - x, image2.shape[1]))
    if ch == 0 or cw == 0:
        return
    alpha = alpha[:ch, :cw]
    image1[y:y + ch, x:x + cw, :] = (image1[y:y + ch, x:x + cw, :] * (1 - alpha) + image2[:ch, :cw, :] * alpha).astype(np.uint8)


def data_loader_worker(tasks_queue, output_queue, quit_workers_event):
    """
    Worker for the data loader.
    """
    cursor_image = cv2.imread(CURSOR_FILE, cv2.IMREAD_UNCHANGED)
    # Assume 16x16
    cursor_image = cursor_image[:16, :16, :]
    cursor_alpha = cursor_image[:, :, 3:] / 255.0
    cursor_image = cursor_image[:, :, :3]

    while True:
        task = tasks_queue.get()
        if task is None:
            break
        trajectory_id, video_path, json_path = task
        video = cv2.VideoCapture(video_path)
        # NOTE: In some recordings, the game seems to start
        #       with attack always down from the beginning, which
        #       is stuck down until player actually presses attack
        # NOTE: It is uncertain if this was the issue with the original code.
        attack_is_stuck = False
        # Scrollwheel is allowed way to change items, but this is
        # not captured by the recorder.
        # Work around this by keeping track of selected hotbar item
        # and updating "hotbar.#" actions when hotbar selection changes.
        # NOTE: It is uncertain is this was/is an issue with the contractor data
        last_hotbar = 0

        with open(json_path) as json_file:
            json_lines = json_file.readlines()
            json_data = "[" + ",".join(json_lines) + "]"
            json_data = json.loads(json_data)
        for i in range(len(json_data)):
            if quit_workers_event.is_set():
                break
            step_data = json_data[i]

            if i == 0:
                # Check if attack will be stuck down
                if step_data["mouse"]["newButtons"] == [0]:
                    attack_is_stuck = True
            elif attack_is_stuck:
                # Check if we press attack down, then it might not be stuck
                if 0 in step_data["mouse"]["newButtons"]:
                    attack_is_stuck = False
            # If still stuck, remove the action
            if attack_is_stuck:
                step_data["mouse"]["buttons"] = [button for button in step_data["mouse"]["buttons"] if button != 0]

            action, is_null_action = json_action_to_env_action(step_data)

            # Update hotbar selection
            current_hotbar = step_data["hotbar"]
            if current_hotbar != last_hotbar:
                action["hotbar.{}".format(current_hotbar + 1)] = 1
            last_hotbar = current_hotbar

            # Read frame even if this is null so we progress forward
            ret, frame = video.read()
            if ret:
                # Skip null actions as done in the VPT paper
                # NOTE: in VPT paper, this was checked _after_ transforming into agent's action-space.
                #       We do this here as well to reduce amount of data sent over.
                if is_null_action:
                    continue
                if step_data["isGuiOpen"]:
                    camera_scaling_factor = frame.shape[0] / MINEREC_ORIGINAL_HEIGHT_PX
                    cursor_x = int(step_data["mouse"]["x"] * camera_scaling_factor)
                    cursor_y = int(step_data["mouse"]["y"] * camera_scaling_factor)
                    composite_images_with_alpha(frame, cursor_image, cursor_alpha, cursor_x, cursor_y)
                cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB, dst=frame)
                frame = np.asarray(np.clip(frame, 0, 255), dtype=np.uint8)
                frame = resize_image(frame, AGENT_RESOLUTION)
                output_queue.put((trajectory_id, frame, action), timeout=QUEUE_TIMEOUT)
            else:
                print(f"Could not read frame from video {video_path}")
        video.release()
        if quit_workers_event.is_set():
            break
    # Tell that we ended
    output_queue.put(None)

class DataLoader:
    """
    Generator class for loading batches from a dataset

    This only returns a single step at a time per worker; no sub-sequences.
    Idea is that you keep track of the model's hidden state and feed that in,
    along with one sample at a time.

    + Simpler loader code
    + Supports lower end hardware
    - Not very efficient (could be faster)
    - No support for sub-sequences
    - Loads up individual files as trajectory files (i.e. if a trajectory is split into multiple files,
      this code will load it up as a separate item).
    """
    def __init__(self, dataset_dir, n_workers=8, batch_size=8, n_epochs=1, max_queue_size=16):
        assert n_workers >= batch_size, "Number of workers must be equal or greater than batch size"
        self.dataset_dir = dataset_dir
        self.n_workers = n_workers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        unique_ids = glob.glob(os.path.join(dataset_dir, "*.mp4"))
        unique_ids = list(set([os.path.basename(x).split(".")[0] for x in unique_ids]))
        self.unique_ids = unique_ids
        # Create tuples of (video_path, json_path) for each unique_id
        demonstration_tuples = []
        for unique_id in unique_ids:
            video_path = os.path.abspath(os.path.join(dataset_dir, unique_id + ".mp4"))
            json_path = os.path.abspath(os.path.join(dataset_dir, unique_id + ".jsonl"))
            demonstration_tuples.append((video_path, json_path))

        assert n_workers <= len(demonstration_tuples), f"n_workers should be lower or equal than number of demonstrations {len(demonstration_tuples)}"

        # Repeat dataset for n_epochs times, shuffling the order for
        # each epoch
        self.demonstration_tuples = []
        for i in range(n_epochs):
            random.shuffle(demonstration_tuples)
            self.demonstration_tuples += demonstration_tuples

        self.task_queue = Queue()
        self.n_steps_processed = 0
        for trajectory_id, task in enumerate(self.demonstration_tuples):
            self.task_queue.put((trajectory_id, *task))
        for _ in range(n_workers):
            self.task_queue.put(None)

        self.output_queues = [Queue(maxsize=max_queue_size) for _ in range(n_workers)]
        self.quit_workers_event = Event()
        self.processes = [
            Process(
                target=data_loader_worker,
                args=(
                    self.task_queue,
                    output_queue,
                    self.quit_workers_event,
                ),
                daemon=True
            )
            for output_queue in self.output_queues
        ]
        for process in self.processes:
            process.start()

    def __iter__(self):
        return self

    def __next__(self):
        batch_frames = []
        batch_actions = []
        batch_episode_id = []

        for i in range(self.batch_size):
            workitem = self.output_queues[self.n_steps_processed % self.n_workers].get(timeout=QUEUE_TIMEOUT)
            if workitem is None:
                # Stop iteration when first worker runs out of work to do.
                # Yes, this has a chance of cutting out a lot of the work,
                # but this ensures batches will remain diverse, instead
                # of having bad ones in the end where potentially
                # one worker outputs all samples to the same batch.
                raise StopIteration()
            trajectory_id, frame, action = workitem
            batch_frames.append(frame)
            batch_actions.append(action)
            batch_episode_id.append(trajectory_id)
            self.n_steps_processed += 1
        return batch_frames, batch_actions, batch_episode_id

    def __del__(self):
        for process in self.processes:
            process.terminate()
            process.join()
