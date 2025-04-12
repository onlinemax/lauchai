# How to run the code


## First you will need to install the following dependencies: 
```bash
pip install -r requirements.txt
```
## Then you'll need to create some folders
```bash
mkdir stocks
mkdir stocks2
mkdir stocks3
```

## Then you need to follow the instructions at this website to install dolt

Follow the instructions at the following url: (dolt installation)[https://docs.dolthub.com/introduction/installation]
## Then you'll be able to get the data by doing this command

```bash
dolt clone post-no-preference/earnings
```
## Then run the sql server by doing those commands
```bash
cd earnings
dolt sql-server -H 127.0.0.1 -P 3306
```
## Finally run the following command on another terminal
```bash
cd ..
python main.py
python firstmonth.py
python merge.py

```

