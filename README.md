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
You can change the verilog design in this .sh
## Run train
```
python parser.py
bash concat.sh
pip install -r requirements.txt
cd GraphSAINT
python setup.py build_ext --inplace
bash run_train.sh
```
