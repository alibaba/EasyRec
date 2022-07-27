# This is a mocked directory, so that processor/test.py 
# could access *_pb2.py files.
# Directly access *_pb2.py from EasyRec does not work
# because processor may use different tensorflow versions
# which leads to conflicts for the underlying tensorflow 
# resources.
