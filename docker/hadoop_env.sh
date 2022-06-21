export PATH=$PATH:$HADOOP_HOME/bin/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$JAVA_HOME/jre/lib/amd64/server:$HADOOP_HOME/lib/native
export CLASSPATH=$($HADOOP_HOME/bin/hadoop classpath --glob)
