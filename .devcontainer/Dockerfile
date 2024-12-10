FROM ubuntu:22.04

ARG USERNAME=tim-external
ARG USER_UID=1001
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to istall software after connecting.
    && apt update \
    && apt install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME


USER $USERNAME
RUN sudo apt install software-properties-common -y
RUN sudo apt install unzip nano net-tools nmap -y
RUN sudo apt install python3 python3-pip -y


RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install osmnx matplotlib geopandas



WORKDIR /home/$USERNAME/Documents
#ENV ROS_DISTRO=humble
ENV SHELL=/bin/bash

RUN echo "PS1='\[\033[1;36m\]\u\[\033[1;31m\]@\[\033[1;32m\]\h:\[\033[1;35m\]\w\[\033[1;31m\]$\[\033[0m\] ' "  >> ~/.bashrc


CMD ["/bin/bash"]


















