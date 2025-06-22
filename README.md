## Clone this reop
```
git clone https://github.com/lorettayao/Cad-contest.git
cd Cad-contest
```
## Replace GraphSAINT
```
rm -rf GraphSAINT
unzip GraphSAINT.zip
mv GraphSAINT/content/GraphSAINT GraphSAINT

```
## Build requirements
```
conda create -n cad_env python=3.8
conda activate cad_env
pip install -r requirements.txt
```
## Run parser

```
python parser.py
```
## Run inference

```
bash run_infer_KUO.sh
```
