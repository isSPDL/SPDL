### Prerequisites

- **Python 3.8** (Error with Python 3.9)

- `pip` version 9.0.1 or higher

If necessary, upgrade your version of `pip`:

```sh
$ python -m pip install --upgrade pip
```

#### gRPC

Install gRPC:

```sh
$ python -m pip install grpcio
```

#### gRPC tools

Python’s gRPC tools include the protocol buffer compiler `protoc` and the special plugin for generating server and client code from `.proto` service definitions. For the first part of our quick-start example, we’ve already generated the server and client stubs from [helloworld.proto](https://github.com/grpc/grpc/tree/v1.38.0/examples/protos/helloworld.proto), but you’ll need the tools for the rest of our quick start, as well as later tutorials and your own projects.

To install gRPC tools, run:

```sh
$ python -m pip install grpcio-tools
```

### Generate gRPC code

```sh
python3 -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. grpc.proto  
```

You can skip this step if "grpc_pb2.py" and  "grpc_pb2_grpc.py" exist.

### PyCryptodome

PyCryptodome is a self-contained Python package of low-level cryptographic primitives.

It supports Python 2.7, Python 3.5 and newer, and PyPy.

You can install it with:

```
pip install pycryptodome
```

### Run!

A simple run with two nodes:

```sh
open terminal_1, run
$ python3 main.py 127.0.0.1:50051 50051
open terminal_2
python3 main.py 127.0.0.1:50053 50053
```

### Generate figure

```sh
$ pip install scipy
$ cd log
$ python3 figure.py 
```

### MNIST
install mnist
```sh
$ pip install mnist
```
parse mnist data set uniformly
```sh
cd mnist_data
python3 mnist_parser
```

