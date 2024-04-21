FROM nvcr.io/nvidia/pytorch:23.03-py3
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.8/dist-packages/torch/lib:/usr/local/lib/python3.8/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64"
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

ARG USER=standard
ARG USER_ID=1001
ARG USER_GROUP=standard
ARG USER_GROUP_ID=1001
ARG USER_HOME=/home/${USER}
# create a user group and a user (this works only for debian based images)
RUN groupadd --gid $USER_GROUP_ID $USER \
    && useradd --uid $USER_ID --gid $USER_GROUP_ID -m $USER

# set container user
USER $USER