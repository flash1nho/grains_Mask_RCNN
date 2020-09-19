# Детектит зерна пшеницы на картинке

# Requirements
- `Python 3.6`

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
- `curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py`
- `python3.6 get-pip.py --force-reinstall`

# Установка Python модулей
- `sudo pip3.6 install -r Mask_RCNN/requirements.txt`

# Download mask rcnn coco and put it in Mask_RCNN folder
- `https://github.com/matterport/Mask_RCNN/releases/download/v1.0/mask_rcnn_coco.h5`

# Train
- `python3.6 train.py`

# Usage
- `python3.6 detect.py --image=<path_to_image>`

# Annotation
- Для создания xml аннотаций: `https://github.com/tzutalin/labelImg`
