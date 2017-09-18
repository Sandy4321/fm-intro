# Startup shell script
# We start with an Anaconda environment for prototyping but will
# refactor for Docker for deployment and sharing

# View Environments available
conda info --envs

# Create a new Anaconda environment
conda create --name fm-intro python=3.5 numpy scikit-learn pandas

# Activate
source activate fm-intro

# Install TensorFlow 1.3.0
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.3.0-py3-none-any.whl


