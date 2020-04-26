# Детектит зерна пшеницы на картинке [WIP]

# Установка ubuntu пакетов
- `sudo apt-get install libssl-dev`
- `sudo apt-get install g++`
- `sudo apt-get install autoconf automake libtool`
- `sudo apt-get install pkg-config`
- `sudo apt-get install cmake`

# Установка Python 3.6.1
- `wget https://www.python.org/ftp/python/3.6.1/Python-3.6.1.tar.xz`
- `tar xvf Python-3.6.1.tar.xz`
- `cd Python-3.6.1`
- `sudo ./configure --enable-optimizations`
- `sudo make altinstall`

# Установка Python модулей
- `sudo pip3.6 install -r Mask_RCNN/requirements.txt`

# Train
- `python3.6 train.py`

# Usage
- `python3.6 detect.py`
