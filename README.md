

# Video-Pre-Training
Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos


> :page_facing_up: [Read Paper](https://cdn.openai.com/vpt/Paper.pdf) \
  :mega: [Blog Post](https://openai.com/blog/vpt) \
  :space_invader: [MineRL Environment](https://github.com/minerllabs/minerl) (note version 1.0+ required) \
  :checkered_flag: [MineRL BASALT Competition](https://www.aicrowd.com/challenges/neurips-2022-minerl-basalt-competition)


# Running agent models

Install pre-requirements for [MineRL](https://minerl.readthedocs.io/en/v1.0.0/tutorials/index.html).
Then install requirements with:

```
pip install git+https://github.com/minerllabs/minerl@v1.0.0
pip install -r requirements.txt
```

To run the code, call

```
python run_agent.py --model [path to .model file] --weights [path to .weight file]
```

After loading up, you should see a window of the agent playing Minecraft.



# Agent Model Zoo
Below are the model files and weights files for various pre-trained Minecraft models.
The 1x, 2x and 3x model files correspond to their respective model weights width.

* [:arrow_down: 1x Model](https://openaipublic.blob.core.windows.net/minecraft-rl/models/foundation-model-1x.model)
* [:arrow_down: 2x Model](https://openaipublic.blob.core.windows.net/minecraft-rl/models/2x.model)
* [:arrow_down: 3x Model](https://openaipublic.blob.core.windows.net/minecraft-rl/models/foundation-model-3x.model)

### Demonstration Only - Behavioral Cloning
These models are trained on video demonstrations of humans playing Minecraft
using behavioral cloning (BC) and are more general than later models which
use reinforcement learning (RL) to further optimize the policy.
Foundational models are trained across all videos in a single training run
while house and early game models refine their respective size foundational
model further using either the housebuilding contractor data or early game video
sub-set. See the paper linked above for more details.

#### Foundational Model :chart_with_upwards_trend:
  * [:arrow_down: 1x Width Weights](https://openaipublic.blob.core.windows.net/minecraft-rl/models/foundation-model-1x.weights)
  * [:arrow_down: 2x Width Weights](https://openaipublic.blob.core.windows.net/minecraft-rl/models/foundation-model-2x.weights)
  * [:arrow_down: 3x Width Weights](https://openaipublic.blob.core.windows.net/minecraft-rl/models/foundation-model-3x.weights)

#### Fine-Tuned from House :chart_with_upwards_trend:
  * [:arrow_down: 3x Width Weights](https://openaipublic.blob.core.windows.net/minecraft-rl/models/bc-house-3x.weights)

#### Fine-Tuned from Early Game :chart_with_upwards_trend:
  * [:arrow_down: 2x Width Weights](https://openaipublic.blob.core.windows.net/minecraft-rl/models/bc-early-game-2x.weights)
  * [:arrow_down: 3x Width Weights](https://openaipublic.blob.core.windows.net/minecraft-rl/models/bc-early-game-3x.weights)

### Models With Environment Interactions
These models further refine the above demonstration based models with a reward
function targeted at obtaining diamond pickaxes. While less general then the behavioral
cloning models, these models have the benefit of interacting with the environment
using a reward function and excel at progressing through the tech tree quickly.
See the paper for more information
on how they were trained and the exact reward schedule.

#### RL from Foundation :chart_with_upwards_trend:
  * [:arrow_down: 2x Width Weights](https://openaipublic.blob.core.windows.net/minecraft-rl/models/rl-from-foundation-2x.weights)

#### RL from House :chart_with_upwards_trend:
  * [:arrow_down: 2x Width Weights](https://openaipublic.blob.core.windows.net/minecraft-rl/models/rl-from-house-2x.weights)

#### RL from Early Game :chart_with_upwards_trend:
  * [:arrow_down: 2x Width Weights](https://openaipublic.blob.core.windows.net/minecraft-rl/models/rl-from-early-game-2x.weights)

# Running Inverse Dynamics Model (IDM)

IDM aims to predict what actions player is taking in a video recording.

Setup:
* Install requirements: `pip install -r requirements.txt`
* Download the IDM model [.model :arrow_down:](https://openaipublic.blob.core.windows.net/minecraft-rl/idm/4x_idm.model) and [.weight :arrow_down:](https://openaipublic.blob.core.windows.net/minecraft-rl/idm/4x_idm.weights) files
* For demonstration purposes, you can use the contractor recordings shared below to. For this demo we use
  [this .mp4](https://openaipublic.blob.core.windows.net/minecraft-rl/data/10.0/cheeky-cornflower-setter-02e496ce4abb-20220421-092639.mp4)
  and [this associated actions file (.jsonl)](https://openaipublic.blob.core.windows.net/minecraft-rl/data/10.0/cheeky-cornflower-setter-02e496ce4abb-20220421-092639.jsonl).

To run the model with above files placed in the root directory of this code:
```
python run_inverse_dynamics_model.py -weights 4x_idm.weights --model 4x_idm.model --video-path cheeky-cornflower-setter-02e496ce4abb-20220421-092639.mp4 --jsonl-path cheeky-cornflower-setter-02e496ce4abb-20220421-092639.jsonl
```

A window should pop up which shows the video frame-by-frame, showing the predicted and true (recorded) actions side-by-side on the left.

Note that `run_inverse_dynamics_model.py` is designed to be a demo of the IDM, not code to put it into practice.

# Using behavioural cloning to fine-tune the models

**Disclaimer:** This code is a rough demonstration only and not an exact recreation of what original VPT paper did (but it contains some preprocessing steps you want to be aware of)! As such, do not expect replicate the original experiments with this code. This code has been designed to be run-able on consumer hardware (e.g., 8GB of VRAM).

Setup:
* Install requirements: `pip install -r requirements.txt`
* Download `.weights` and `.model` file for model you want to fine-tune.
* Download contractor data (below) and place the `.mp4` and `.jsonl` files to the same directory (e.g., `data`). With default settings, you need at least 12 recordings.

If you downloaded the "1x Width" models and placed some data under `data` directory, you can perform finetuning with

```
python behavioural_cloning.py --data-dir data --in-model foundation-model-1x.model --in-weights foundation-model-1x.weights --out-weights finetuned-1x.weights
```

You can then use `finetuned-1x.weights` when running the agent. You can change the training settings at the top of `behavioural_cloning.py`.

Major limitations:
- Only trains single step at the time, i.e., errors are not propagated through timesteps.
- Computes gradients one sample at a time to keep memory use low, but also slows down the code.

# Contractor Demonstrations

### Versions
Over the course of the project we requested various demonstrations from contractors
which we release as index files below. In general, major recorder versions change for a new
prompt or recording feature while bug-fixes were represented as minor version changes.
However, some
recorder versions we asked contractors to change their username when recording particular
modalities. Also, as contractors internally ask questions, clarification from one contractor may
result in a behavioral change in the other contractor. It is intractable to share every contractor's
view for each version, but we've shared the prompts and major clarifications for each recorder
version where the task changed significantly.

  <details>
  <summary>Initial Prompt</summary>

  We are collecting data for training AI models in Minecraft. You'll need to install java, download the modified version of minecraft (that collects and uploads your play data), and play minecraft survival mode! Paid per hour of gameplay. Prior experience in minecraft not. necessary. We do not collect any data that is unrelated to minecraft from your computer.

  </details>

The following is a list of the available versions:

* **6.x** Core recorder features subject to change [:arrow_down: index file](https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_6xx_Jun_29.json)
  * 6.9 First feature complete recorder version
  * 6.10 Fixes mouse scaling on Mac when gui is open
  * 6.11 Tracks the hotbar slot
  * 6.13 Sprinting, swap-hands, ... (see commits below)
    <details>
    <summary>Commits</summary>

    * improve replays that are cut in the middle of gui; working on riding boats / replays cut in the middle of a run
    * improve replays by adding dwheel action etc, also, loosen up replay tolerances
    * opencv version bump
    * add swap hands, and recording of the step timestamp
    * implement replaying from running and sprinting and tests
    * do not record sprinting (can use stats for that)
    * check for mouse button number, ignore >2
    * handle the errors when mouse / keyboard are recorded as null

    </details>
* **7.x** Prompt changes [:arrow_down: index file](https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_7xx_Apr_6.json)
  * 7.6 Bump version for internal tracking
    <details>
    <summary>Additional ask to contractors</summary>

    Right now, early game data is especially valuable to us. As such, we request that at least half of the data you upload is from the first 30 minutes of the game. This means that, for every hour of gameplay you spend in an older world, we ask you to play two sessions in which you create a new world and play for 30 minutes. You can play for longer in these worlds, but only the first 30 minutes counts as early game data.

    </details>
* **8.x** :clipboard: House Building from Scratch Task [:arrow_down: index](https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_8xx_Jun_29.json)
  <details>
  <summary>Changes and Prompt</summary>

  Hi all! Thank you for your hard work so far.

  This week we would like to have you all collect data on a specific task.

  This comes with a new recorder version 8.0 which you will need to update your recording script to download.

  This week we would like you to use a new world each time you play, so loading existing worlds is disabled.

  The new task is as follows:

  Starting in a new world, build a simple house in 10-15 minutes. This corresponds to one day and a bit of the night. Please use primarily wood, dirt, and sand, as well as crafted wood items such as doors, fences, ect. in constructing your house. Avoid using difficult items such as stone. Aside from those constraints, you may decorate the structure you build as you wish. It does not need to have any specific furniture. For example, it is OK if there is no bed in your house. If you have not finished the house by the sunrise (20 minutes) please exit and continue to another demonstration. Please continue to narrate what you are doing while completing this task.

  Since you will be unable to resume building after exiting Minecraft or going back to the main menu, you must finish these demonstrations in one session. Pausing via the menu is still supported. If you want to view your creations later, they will be saved locally so you can look at them in your own time. We may use these save files in a future task so if you have space, please leave the save files titled “build-house-15-min-“.

  For this week try to avoid all cobblestone / stone / granite

  For this week we just want simple houses without sleeping. If 10 minutes is too short, let us know and we can think of how to adjust!

  Stone tools are ok but I think you may run-out of time

  Changes:
    * Timer ends episode after 10 realtime minutes
    * Worlds are named: `"build-house-15-min-" + Math.abs(random.nextInt());`

  </details>

  * Note this version introduces 10-minute timer that ends the episode. It
  cut experiments short occasionally and was fixed in 9.1
  * 8.0 Simple House
  * 8.2 Update upload script
* **9.x** :clipboard: House Building from Random Starting Materials Task [:arrow_down: index](https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_9xx_Jun_29.json)
    <details>
    <summary>Changes and Prompt</summary>

    You now will have 10 minutes to use the provided resources to build your house / home / or structure. In this version, the experiment will time out after 10 minutes if you are not complete so don't be alarmed if that happens, it is intentional.

    No need to use up all the resources! It's ok to collect a few things but spend the majority of the time placing blocks (the act of placing seems to be harder to learn)

    Changes:
    * Worlds are named: `"design-house-10-min-" + Math.abs(random.nextInt());`
    * Starting inventory given by code below
    </details>

    <details>
    <summary>Random Starting Inventory Code</summary>

  ```java
        Random random = new Random();
        List<ItemStack> hotbar = new ArrayList<>();
        List<ItemStack> inventory = new ArrayList<>();

        // Ensure we give the player the basic tools in their hot bar
        hotbar.add(new ItemStack(Items.STONE_AXE));
        hotbar.add(new ItemStack(Items.STONE_PICKAXE));
        hotbar.add(new ItemStack(Items.STONE_SHOVEL));
        hotbar.add(new ItemStack(Items.CRAFTING_TABLE));

        // Add some random items to the player hotbar as well
        addToList(hotbar, inventory, Items.TORCH, random.nextInt(16) * 2 + 2);

        // Next add main building blocks
        if (random.nextFloat() < 0.7) {
           addToList(hotbar, inventory, Items.OAK_FENCE_GATE, random.nextInt(5));
           addToList(hotbar, inventory, Items.OAK_FENCE, random.nextInt(5) * 64);
           addToList(hotbar, inventory, Items.OAK_DOOR, random.nextInt(5));
           addToList(hotbar, inventory, Items.OAK_TRAPDOOR, random.nextInt(2) * 2);
           addToList(hotbar, inventory, Items.OAK_PLANKS, random.nextInt(3) * 64 + 128);
           addToList(hotbar, inventory, Items.OAK_SLAB, random.nextInt(3) * 64);
           addToList(hotbar, inventory, Items.OAK_STAIRS, random.nextInt(3) * 64);
           addToList(hotbar, inventory, Items.OAK_LOG, random.nextInt(2) * 32);
           addToList(hotbar, inventory, Items.OAK_PRESSURE_PLATE, random.nextInt(5));
        } else {
           addToList(hotbar, inventory, Items.BIRCH_FENCE_GATE, random.nextInt(5));
           addToList(hotbar, inventory, Items.BIRCH_FENCE, random.nextInt(5) * 64);
           addToList(hotbar, inventory, Items.BIRCH_DOOR, random.nextInt(5));
           addToList(hotbar, inventory, Items.BIRCH_TRAPDOOR, random.nextInt(2) * 2);
           addToList(hotbar, inventory, Items.BIRCH_PLANKS, random.nextInt(3) * 64 + 128);
           addToList(hotbar, inventory, Items.BIRCH_SLAB, random.nextInt(3) * 64);
           addToList(hotbar, inventory, Items.BIRCH_STAIRS, random.nextInt(3) * 64);
           addToList(hotbar, inventory, Items.BIRCH_LOG, random.nextInt(2) * 32);
           addToList(hotbar, inventory, Items.BIRCH_PRESSURE_PLATE, random.nextInt(5));
        }

        // Now add some random decoration items to the player inventory
        addToList(hotbar, inventory, Items.CHEST, random.nextInt(3));
        addToList(hotbar, inventory, Items.FURNACE, random.nextInt(2) + 1);
        addToList(hotbar, inventory, Items.GLASS_PANE,  random.nextInt(5) * 4);
        addToList(hotbar, inventory, Items.WHITE_BED, (int) (random.nextFloat() + 0.2)); // Bed 20% of the time
        addToList(hotbar, inventory, Items.PAINTING, (int) (random.nextFloat() + 0.1)); // Painting 10% of the time
        addToList(hotbar, inventory, Items.FLOWER_POT, (int) (random.nextFloat() + 0.1) * 4); // 4 Flower pots 10% of the time
        addToList(hotbar, inventory, Items.OXEYE_DAISY, (int) (random.nextFloat() + 0.1) * 4); // 4 Oxeye daisies 10% of the time
        addToList(hotbar, inventory, Items.POPPY, (int) (random.nextFloat() + 0.1) * 4); // 4 Poppies 10% of the time
        addToList(hotbar, inventory, Items.SUNFLOWER, (int) (random.nextFloat() + 0.1) * 4); // 4 Sunflowers 10% of the time

        // Shuffle the hotbar slots and inventory slots
        Collections.shuffle(hotbar);
        Collections.shuffle(inventory);

        // Give the player the items
        this.mc.getIntegratedServer().getPlayerList().getPlayers().forEach(p -> {
           if (p.getUniqueID().equals(this.getUniqueID())) {
               hotbar.forEach(p.inventory::addItemStackToInventory);
               inventory.forEach(p.inventory::addItemStackToInventory);
           }
        });
  ```

    </details>

     * 9.0 First version
     * 9.1 Fixed timer bug
* **10.0** :clipboard: Obtain Diamond Pickaxe Task [:arrow_down: index](https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_10xx_Jun_29.json)
  <details>
  <summary>Changes and Prompt</summary>
  Prompt:

  For this new task we have given you 20 minutes to craft a diamond pickaxe. We ask that you do not try to search for villages or other ways of getting diamonds, but if you are spawned in view of one, or happen to fall into a cave structure feel free to explore it for diamonds.
  If 20 min is not enough that is OK. It will happen on some seeds because of bad luck. Please do not use glitches to find the diamonds.

  Changes:
  * change to 20 minute time limit
  * _don't count gui time as part of the time limit_
  * World are named `"collect-diamond-pickaxe-15min-" + Math.abs(random.nextInt());`

  </details>


Sometimes we asked the contractors to signify other tasks besides changing the version. This
primarily occurred in versions 6 and 7 as 8, 9 and 10 are all task specific.

<details>
<summary>Prompt to contractors (click to show)</summary>
Another request about additional time - please use some of it to chop trees. Specifically, please start the recorder by adding --username treechop argument to the script (i.e. use play --username treechop on windows, ./play.sh --username treechop on osx/linux), and spend some time chopping trees! Getting wooden or stone tools is ok, but please spend the majority of the with username treechop specifically chopping. I did it myself for about 15 minutes, and it does get boring pretty quickly, so I don't expect you to do it all the time, but please do at least a little bit of chopping. Feel free to play normally the rest of the time (but please restart without --username treechop argument when you are not chopping)
However, it is preferable that you start a new world though, and use only the tools that are easily obtainable in that world. I'll see what I can do about getting player an iron axe - that sounds reasonable, and should not be hard, but will require a code update.
</details>

### Environment
We restrict the contractors to playing Minecraft in windowed mode at 720p which we downsample at 20hz to 360p
to minimize space. We also disabled the options screen to prevent the contractor from
changing things such as brightness, or rendering options. We ask contractors not to press keys
such as f3 which shows a debug overlay, however some contractors may still do this.


### Data format

Demonstrations are broken up into up to 5 minute segments consisting of a series of
compressed screen observations, actions, environment statistics, and a checkpoint
save file from the start of the segment. Each relative path in the index will
have all the files for that given segment, however if a file was dropped while
uploading, the corresponding relative path is not included in the index therefore
there may be missing chunks from otherwise continuous demonstrations.

Index files are provided for each version as a json file:
```json
{
  "basedir": "https://openaipublic.blob.core.windows.net/data/",
  "relpaths": [
    "8.0/cheeky-cornflower-setter-74ae6c2eae2e-20220315-122354",
    ...
  ]
}
```
Relative paths follow the following format:
* `<recorder-version>/<contractor-alias>-<session-id>-<date>-<time>`

> Note that due to network errors, some segments may be missing from otherwise
continuous demonstrations.

Your data loader can then find following files:
* Video observation: `<basedir>/<relpath>.mp4`
* Action file: `<basedir>/<relpath>.jsonl`
* Options file: `<basedir>/<relpath>-options.json`
* Checkpoint save file: `<basedir>/<relpath>.zip`

The action file is **not**  a valid json object: each line in
action file is an individual action dictionary.

For v7.x, the actions are in form
```json
{
  "mouse": {
    "x": 274.0,
    "y": 338.0,
    "dx": 0.0,
    "dy": 0.0,
    "scaledX": -366.0,
    "scaledY": -22.0,
    "dwheel": 0.0,
    "buttons": [],
    "newButtons": []
  },
  "keyboard": {
    "keys": [
      "key.keyboard.a",
      "key.keyboard.s"
    ],
    "newKeys": [],
    "chars": ""
  },
  "isGuiOpen": false,
  "isGuiInventory": false,
  "hotbar": 4,
  "yaw": -112.35006,
  "pitch": 8.099996,
  "xpos": 841.364694513396,
  "ypos": 63.0,
  "zpos": 24.956354839537802,
  "tick": 0,
  "milli": 1649575088006,
  "inventory": [
    {
      "type": "oak_door",
      "quantity": 3
    },
    {
      "type": "oak_planks",
      "quantity": 59
    },
    {
      "type": "stone_pickaxe",
      "quantity": 1
    },
    {
      "type": "oak_planks",
      "quantity": 64
    }
  ],
  "serverTick": 6001,
  "serverTickDurationMs": 36.3466,
  "stats": {
    "minecraft.custom:minecraft.jump": 4,
    "minecraft.custom:minecraft.time_since_rest": 5999,
    "minecraft.custom:minecraft.play_one_minute": 5999,
    "minecraft.custom:minecraft.time_since_death": 5999,
    "minecraft.custom:minecraft.walk_one_cm": 7554,
    "minecraft.use_item:minecraft.oak_planks": 5,
    "minecraft.custom:minecraft.fall_one_cm": 269,
    "minecraft.use_item:minecraft.glass_pane": 3
  }
}
```

# BASALT 2022 dataset

We also collected a dataset of demonstrations for the [MineRL BASALT 2022](https://www.aicrowd.com/challenges/neurips-2022-minerl-basalt-competition) competition, with around 150GB of data per task.

**Note**: To avoid confusion with the competition rules, the action files (.jsonl) have been stripped of information that is not allowed in the competition. We will upload unmodified dataset after the competition ends.

* **FindCave** [:arrow_down: index file](https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/find-cave-Jul-28.json)
  * <details>
    <summary>Prompt to contractors (click to show)</summary>

    ```
    Look around for a cave. When you are inside one, quit the game by opening main menu and pressing "Save and Quit To Title".
    You are not allowed to dig down from the surface to find a cave.

    Timelimit: 3 minutes.
    Example recordings: https://www.youtube.com/watch?v=TclP_ozH-eg
    ```
    </details>
* **MakeWaterfall** [:arrow_down: index file](https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/waterfall-Jul-28.json)
  * <details>
    <summary>Prompt to contractors (click to show)</summary>

    ```
    After spawning in a mountainous area with a water bucket and various tools, build a beautiful waterfall and then reposition yourself to “take a scenic picture” of the same waterfall, and then quit the game by opening the menu and selecting "Save and Quit to Title"

    Timelimit: 5 minutes.
    Example recordings: https://youtu.be/NONcbS85NLA
    ```
    </details>
* **MakeVillageAnimalPen** [:arrow_down: index file](https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/pen-animals-Jul-28.json)
  * <details>
    <summary>Prompt to contractors (click to show)</summary>

    ```
    After spawning in a village, build an animal pen next to one of the houses in a village. Use your fence posts to build one animal pen that contains at least two of the same animal. (You are only allowed to pen chickens, cows, pigs, sheep or rabbits.) There should be at least one gate that allows players to enter and exit easily. The animal pen should not contain more than one type of animal. (You may kill any extra types of animals that accidentally got into the pen.) Don’t harm the village.
    After you are done, quit the game by opening the menu and pressing "Save and Quit to Title".

    You may need to terraform the area around a house to build a pen. When we say not to harm the village, examples include taking animals from existing pens, damaging existing houses or farms, and attacking villagers. Animal pens must have a single type of animal: pigs, cows, sheep, chicken or rabbits.

    The food items can be used to lure in the animals: if you hold seeds in your hand, this attracts nearby chickens to you, for example.

    Timelimit: 5 minutes.
    Example recordings: https://youtu.be/SLO7sep7BO8
    ```
    </details>
* **BuildVillageHouse** [:arrow_down: index file](https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/build-house-Jul-28.json)
  * <details>
    <summary>Prompt to contractors (click to show)</summary>

    ```
    Taking advantage of the items in your inventory, build a new house in the style of the village (random biome), in an appropriate location (e.g. next to the path through the village), without harming the village in the process.
    Then give a brief tour of the house (i.e. spin around slowly such that all of the walls and the roof are visible).

    * You start with a stone pickaxe and a stone axe, and various building blocks. It’s okay to break items that you misplaced (e.g. use the stone pickaxe to break cobblestone blocks).
    * You are allowed to craft new blocks.

    Please spend less than ten minutes constructing your house.

    You don’t need to copy another house in the village exactly (in fact, we’re more interested in having slight deviations, while keeping the same "style"). You may need to terraform the area to make space for a new house.
    When we say not to harm the village, examples include taking animals from existing pens, damaging existing houses or farms, and attacking villagers.

    After you are done, quit the game by opening the menu and pressing "Save and Quit to Title".

    Timelimit: 12 minutes.
    Example recordings: https://youtu.be/WeVqQN96V_g
    ```
    </details>



# Contribution
This was a large effort by a dedicated team at OpenAI:
[Bowen Baker](https://github.com/bowenbaker),
[Ilge Akkaya](https://github.com/ilge),
[Peter Zhokhov](https://github.com/pzhokhov),
[Joost Huizinga](https://github.com/JoostHuizinga),
[Jie Tang](https://github.com/jietang),
[Adrien Ecoffet](https://github.com/AdrienLE),
[Brandon Houghton](https://github.com/brandonhoughton),
[Raul Sampedro](https://github.com/samraul),
Jeff Clune
The code here represents a minimal version of our model code which was
prepared by [Anssi Kanervisto](https://github.com/miffyli) and others so that these models could be used as
part of the MineRL BASALT competition.
