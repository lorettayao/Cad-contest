## Clone this repo
```
git clone https://github.com/lorettayao/Cad-contest.git
cd Cad-contest
```
## Replace GraphSAINT
```
rm -rf GraphSAINT
unzip GraphSAINT.zip
```
## Build requirements
```
conda create -n cad_env python==3.8
conda activate cad_env
pip3 install -r requirements.txt
```
## Run parser

```
python3 parser.py
```
## Run inference
```
bash run_infer_KUO.sh
```
You can change the verilog design in this .sh by changing the design_name.
## Run train
```
python3 parser.py
cd concat_train
bash concat.sh
cd ..
```

(pip3 install -r requirements.txt, only if you have not done it before)
```
cd GraphSAINT
python3 graphsaint/setup.py build_ext --inplace
```
(Change config in GraphSAINT/train_config/DATE21.yml if you want)
```
bash run_train.sh
```
