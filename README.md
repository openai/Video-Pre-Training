

# Video-Pre-Training
Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos


> :page_facing_up: [Read Paper](https://cdn.openai.com/vpt/Paper.pdf) \
  :mega: [Blog Post](https://openai.com/blog/vpt) \
  :space_invader: [MineRL Environment](https://github.com/minerllabs/minerl) (note version 1.0+ required) \
  :checkered_flag: [MineRL BASALT Competition](https://www.aicrowd.com/challenges/neurips-2022-minerl-basalt-competition)


## Running models

Install requirements with:

```
pip install git+https://github.com/minerllabs/minerl@v1.0.0
pip install -r requirements.txt
```

To run the code, call

```
python run_agent.py --model [path to .model file] --weights [path to .weight file]
```

After loading up, you should see a window of the agent playing Minecraft.



## Model Zoo
Below are the model files and weights files for various pre-trained Minecraft models.
The 1x, 2x and 3x model files correspond to their respective model weights width. 

* [:arrow_down: 1x Model](https://openaipublic.blob.core.windows.net/minecraft-rl/models/1x.model)
* [:arrow_down: 2x Model](https://openaipublic.blob.core.windows.net/minecraft-rl/models/2x.model)
* [:arrow_down: 3x Model](https://openaipublic.blob.core.windows.net/minecraft-rl/models/3x.model)

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

## Contractor Demonstrations Dataset
We are releasing contractor data collected over the course of the project. Links to index 
files with more information will be linked here as the data released.


Currently, there is no contractor data available for download at this time


## Contribution
This was a large effort by a dedicated team at OpenAI:
[Bowen Baker](https://github.com/bowenbaker), 
[Ilge Akkaya](https://github.com/ilge), 
[Peter Zhokhov](https://github.com/pzhokhov), 
[Joost Huizinga](https://github.com/JoostHuizinga), 
[Jie Tang](https://github.com/jietang), 
[Adrien Ecoffet](https://github.com/AdrienLE),
[Brandon Houghton](https://github.com/brandonhoughton), 
[Raul Sampedro](https://github.com/raul-openai), 
Jeff Clune 
The code here represents a minimal version of our model code which was 
prepared by [Anssi Kanervisto](https://github.com/miffyli) and others so that these models could be used as 
part of the MineRL BASALT competition. 
