echo "START startup-hook.sh yuhu!!"
# YOLOv5 requirements
# Usage: pip install -r requirements.txt
activate

# Base ----------------------------------------
#pip install matplotlib>=3.2.2
conda install -y -c conda-forge matplotlib
#pip install numpy>=1.18.5
conda install -y -c anaconda numpy
#pip install opencv-python>=4.1.1
conda install -y -c conda-forge opencv
#pip install Pillow>=7.1.2
conda install -y -c conda-forge pillow
#pip install PyYAML>=5.3.1
conda install -y -c conda-forge pyyaml
#pip install requests>=2.23.0
conda install -y -c conda-forge requests
#pip install scipy>=1.4.1
conda install -y -c conda-forge scipy
#pip install torch>=1.7.0
#pip install torchvision>=0.8.1
#pip install tqdm>=4.64.0
conda install -y -c conda-forge tqdm
#pip install protobuf
conda install -y -c conda-forge protobuf

# Logging -------------------------------------
#pip install tensorboard>=2.4.1
conda install -y -c conda-forge tensorboard

# Plotting ------------------------------------
#pip install pandas>=1.1.4
conda install -y -c conda-forge pandas
#pip install seaborn>=0.11.0
conda install -y -c conda-forge seaborn
#pip install --upgrade scikit-image
conda install -y -c conda-forge scikit-image

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
#pip install ipython  # interactive notebook
conda install -y -c conda-forge ipython
#pip install psutil  # system utilization
conda install -y -c conda-forge psutil
pip install thop>=0.1.1  # FLOPs computation
#pip install albumentations>=1.0.3
conda install -y -c conda-forge albumentations
#pip install pycocotools>=2.0  # COCO mAP
conda install -y -c conda-forge pycocotools
# roboflow

conda install -y -c conda-forge python-spams
conda install -y libgcc

#pip install spams
#apt-get -y install libblas-dev liblapack-dev gfortran
#apt-get -y install libblas-dev liblapack-dev
#pip install Cython
#pip install --upgrade numpy
#pip install mkl
#pip install setuptools
#pip install python-spams scipy

#apt-get -y install libblas-dev liblapack-dev gfortran
#pip install mkl
#pip install spams
#pip install spams-bin
#conda install -y -c conda-forge matplotlib
#conda install -y -c conda-forge python-spams
#conda update -y libstdcxx-ng
#conda install -y -c anaconda scipy
#conda install -y libgcc gmp
