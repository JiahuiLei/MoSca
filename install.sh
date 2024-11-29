ENV_NAME=mosca
NUMPY_VERSION=1.26.4

conda remove -n $ENV_NAME --all -y
conda create -n $ENV_NAME gcc_linux-64=9 gxx_linux-64=9 python=3.10 numpy=$NUMPY_VERSION -y

source activate $ENV_NAME

which python
which pip

CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
CPP=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
$CC --version
$CXX --version

################################################################################    
pip install numpy==$NUMPY_VERSION
conda install pytorch==2.1.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
# conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install fvcore iopath -c fvcore -c iopath -c conda-forge -y
conda install nvidiacub -c bottler -y
conda install pytorch3d -c pytorch3d -y
pip install pyg_lib torch_scatter torch_geometric torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
################################################################################

################################################################################
echo "Install other dependencies..."
conda install xformers -c xformers -y
pip install -r requirements.txt
pip install numpy==$NUMPY_VERSION
################################################################################

################################################################################
echo "Install GS..."
pip install lib_render/simple-knn
pip install lib_render/diff-gaussian-rasterization-alphadep-add3
pip install lib_render/diff-gaussian-rasterization-alphadep
pip install lib_render/gof-diff-gaussian-rasterization
################################################################################

################################################################################
pip install numpy==$NUMPY_VERSION
pip install -U scikit-learn 
pip install -U scipy
pip install opencv-python==4.10.0.84
pip install mmcv-full==1.7.2
################################################################################

################################################################################
echo "Install JAX for evaluating DyCheck"
pip install -r jax_requirements.txt
################################################################################