# CloseAirCombat
An environment based on JSBSIM aimed at one-to-one close air combat.


### Install 

```shell
# create python env
conda create -n jsbsim python=3.8
# install pytorch
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
# install dependency
pip install jsbsim==1.1.6 geographiclib gym wandb icecream setproctitle matplotlib
```
- Download Shapely‑1.7.1‑cp38‑cp38‑win_amd64.whl from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely), and `pip install shaply` from local file.

### Train

cd scripts
bash train_jsbsim.sh
