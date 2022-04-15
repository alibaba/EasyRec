set(PYTHON "python")

execute_process(
    COMMAND
        ${PYTHON} -c
        "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()))"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TF_COMPILE_FLAGS)

execute_process(
    COMMAND
        ${PYTHON} -c
        "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()))"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TF_LINK_FLAGS)

execute_process(
    COMMAND
        ${PYTHON} -c
        "import tensorflow as tf; print(tf.__version__)"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TF_VERSION)
string(REPLACE "." ";" TF_VERSION_LIST ${TF_VERSION})
list(GET TF_VERSION_LIST 0 TF_VERSION_MAJOR)
list(GET TF_VERSION_LIST 1 TF_VERSION_MINOR)
list(GET TF_VERSION_LIST 2 TF_VERSION_PATCH)

message(STATUS "TensorFlow version = ${TF_VERSION}")

string(REGEX MATCH "(^-L.*\ )" TF_LINK_DIR ${TF_LINK_FLAGS})
string(REPLACE "-L" "" TF_LINK_DIR ${TF_LINK_DIR})
string(REPLACE " " "" TF_LINK_DIR ${TF_LINK_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorFlow 
    DEFAULT_MSG TF_LINK_DIR
)

if (TensorFlow_FOUND)
    add_definitions(-DEIGEN_USE_GPU)
    mark_as_advanced(TF_LINK_DIR TF_LINK_FLAGS TF_VERSION TF_VERSION_MAJOR TF_VERSION_MINOR TF_VERSION_PATCH)
    add_definitions(-DTF_VERSION_MAJOR=${TF_VERSION_MAJOR})
    message(STATUS "TF LINK FLAGS = ${TF_LINK_FLAGS}")
    message(STATUS "TF link dir = ${TF_LINK_DIR}")
    message(STATUS "TF COMPILE FLAGS = ${TF_COMPILE_FLAGS}")
endif()