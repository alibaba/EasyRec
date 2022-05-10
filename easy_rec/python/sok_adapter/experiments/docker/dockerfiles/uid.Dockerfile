FROM easyrec_enhanced:latest
#FROM easyrec_deeprec_enhanced:latest

ARG DOCKER_USER
ARG USER_UID
RUN useradd -m $DOCKER_USER -u ${USER_UID} -s /bin/bash
# passwd is docker
RUN echo "$DOCKER_USER:docker" | chpasswd && adduser $DOCKER_USER sudo