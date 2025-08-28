# GS_HS
本项目的目标是探索如何在仿真环境中可靠地评估真实世界的机器人操作策略
# Installation

```
conda create -n env python=3.10.9 
conda activate env 
pip install -r requirements.txt

cd ./submodules/pyroboplan
pip install -e .
```
CHECK NUMPY'S VERSION !!!
```
pip uninstall numpy
pip install numpy==1.26.4 
```
U can ignore this ERROR
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
cmeel-boost 1.87.0.1 requires numpy<2.4,>=2.2; python_version >= "3.10.0", but you have numpy 1.26.4 which is incompatible.
```
# Some Modification
Issue 1:
```
File "/home/ubuntu/anaconda3/envs/env/lib/python3.10/site-packages/serial/model.py", line 11, in <module>
    from collections import OrderedDict, Callable, Set, Sequence
ImportError: cannot import name 'Callable' from 'collections' (/home/ubuntu/anaconda3/envs/env/lib/python3.10/collections/__init__.py)
```
solustion:
```
from collections.abc import Callable, Set, Sequence
from collections import OrderedDict
```
Issue 2:
```
File "/home/ubuntu/anaconda3/envs/env/lib/python3.10/site-packages/serial/properties.py", line 9, in <module>
from collections import Mapping, Sequence, Set, Iterable, Callable
ImportError: cannot import name 'Mapping' from 'collections' (/home/ubuntu/anaconda3/envs/env/lib/python3.10/collections/__init__.py)
```
solustion:
```
from collections.abc import Mapping, Sequence, Set, Iterable, Callable
```
Issue 3:
```
File "/home/ubuntu/anaconda3/envs/env/lib/python3.10/site-packages/serial/meta.py", line 9, in <module>
    from collections import Callable, OrderedDict
ImportError: cannot import name 'Callable' from 'collections' (/home/ubuntu/anaconda3/envs/env/lib/python3.10/collections/__init__.py)
```
solustion:
```
from collections.abc import Callable
from collections import OrderedDict
```
# Mode Configuation
```
lerobot/common/robots/sim_single_piper/sim_single_piper.py
```
```
self.cfg = EasyDict({
            # change to your scene's path
            "mujoco_config": {
                "model_path": "/home/ubuntu/GS_HS/model_assets/piper_on_desk/scene.xml",},
            # set as you like
            "sim_mode": "record",
            # "sim_mode": "inference",
            "camera_names": ["wrist_cam", "3rd"],
        })
```