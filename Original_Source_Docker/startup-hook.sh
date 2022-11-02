echo "START startup-hook.sh yuhu!"
# YOLOv5 requirements
# Usage: pip install -r requirements.txt

#apt-get install libtcmalloc-minimal4
#export LD_PRELOAD="/usr/lib/libtcmalloc_minimal.so.4"


#export DEBIAN_FRONTEND=noninteractive
#apt-get install -y tzdata
#ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime
#dpkg-reconfigure --frontend noninteractive tzdata
#pip install mkl
#pip install -U pip setuptools
#apt install -y git libblas-dev liblapack-dev gfortran
#export SETUPTOOLS_ENABLE_FEATURES="legacy-editable"
#git clone https://github.com/getspams/spams-python
#cd spams-python
#pip install -e .
#cd ..


#
#apt -y update
#apt -y upgrade
#apt install -y git python3-dev python3-virtualenv python3-tk gcc g++ libeigen3-dev r-base r-cran-randomfields libicu-dev libgmp-dev libmpfr-dev libcgal-dev gmsh libfreetype6-dev libxml2-dev libxslt-dev libblas-dev liblapack-dev gfortran
#echo "START startup-hook.sh yuhu!1111111111"
#pip install mkl

#echo "START startup-hook.sh yuhu!2222222222"
#pip install spams
#pip install python-spams
#echo "START startup-hook.sh yuhu!3333333333"



#pip install tiatoolbox

# Base ----------------------------------------
pip install matplotlib>=3.2.2
pip install numpy>=1.18.5
pip install opencv-python>=4.1.1
pip install Pillow>=7.1.2
pip install PyYAML>=5.3.1
pip install requests>=2.23.0
pip install scipy>=1.4.1
pip install torch>=1.7.0
pip install torchvision>=0.8.1
pip install tqdm>=4.64.0
pip install protobuf

# Logging -------------------------------------
pip install tensorboard>=2.4.1

# Plotting ------------------------------------
pip install pandas>=1.1.4
pip install seaborn>=0.11.0
pip install --upgrade scikit-image

# Export --------------------------------------
# coremltools>=5.2  # CoreML export
# onnx>=1.9.0  # ONNX export
# onnx-simplifier>=0.4.1  # ONNX simplifier
# nvidia-pyindex  # TensorRT export
# nvidia-tensorrt  # TensorRT export
# scikit-learn==0.19.2  # CoreML quantization
# tensorflow>=2.4.1  # TFLite export (or tensorflow-cpu, tensorflow-aarch64)
# tensorflowjs>=3.9.0  # TF.js export
# openvino-dev  # OpenVINO export

# Extras --------------------------------------
pip install ipython  # interactive notebook
pip install psutil  # system utilization
pip install thop>=0.1.1  # FLOPs computation
pip install albumentations>=1.0.3
pip install pycocotools>=2.0  # COCO mAP
# roboflow

