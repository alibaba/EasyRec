set -e

# -------- get TF version -------------- #
TfVersion=`python3 -c "import tensorflow as tf; print(tf.__version__.strip().split('.'))"`
TfMajor=`python3 -c "print($TfVersion[0])"`
TfMinor=`python3 -c "print($TfVersion[1])"`

if [ "$TfMajor" -eq 2 ]; then
    cd tf2/
    bash script.sh; exit 0;
else
    cd tf1/
    bash script.sh; exit 0;
fi
