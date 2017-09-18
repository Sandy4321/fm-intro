# Startup shell script
# We start with an Anaconda environment for prototyping but will
# refactor for Docker for deployment and sharing
# === Windows GPU

# View Environments available
conda info --envs

# Create a new Anaconda environment
conda create --name fm-intro python=3.5 numpy scikit-learn pandas

# Activate
activate fm-intro

# Install TensorFlow 1.3.0
pip install --ignore-installed --upgrade tensorflow-gpu


# Remember to install CUDA Cudnn
# Once you do that, you must add them to the PATH environments. For example:
# For CUDA
#	 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin
#	 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\libnvvp
# Extract Cudnn and point to it here:
#	 C:\cuda
#	 C:\cuda\bin
