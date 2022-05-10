FROM deeprec_enhanced:latest

# install deeprec wheel
ADD ./tensorflow-1.15.5+deeprec2201-cp36-cp36m-linux_x86_64.whl /tmp/.
RUN env PATH=$PATH pip3 uninstall -y tensorflow tensorboard tensorflow-estimator
RUN env PATH=$PATH pip3 install /tmp/tensorflow-1.15.5+deeprec2201-cp36-cp36m-linux_x86_64.whl

# Install pip
ADD ./requirements /tmp/requirements
RUN pip3 install -r /tmp/requirements/runtime.txt
RUN pip3 install -r /tmp/requirements/tests.txt