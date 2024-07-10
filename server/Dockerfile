FROM nvcr.io/nvidia/pytorch:24.03-py3

RUN mkdir /TGI

RUN apt-get update

WORKDIR /TGI

COPY . /TGI

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    curl \
    unzip

# Install Rust using rustup
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Set environment variables for Rust
ENV PATH="/root/.cargo/bin:${PATH}"

# Verify Rust installation by checking the rustc version
RUN rustc --version || { echo "Error occured while installing Rust"; exit 1; }

# Set environment variables for protoc
ENV PROTOC_ZIP=protoc-21.12-linux-x86_64.zip

# Download and install protoc
RUN curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP && \
    unzip -o $PROTOC_ZIP -d /usr/local bin/protoc && \
    unzip -o $PROTOC_ZIP -d /usr/local 'include/*' && \
    rm -f $PROTOC_ZIP


RUN cd text-generation-inference && BUILD_EXTENSIONS=True make install

# install dependencies for rust
RUN apt-get install libssl-dev gcc -y

RUN apt-get update && apt-get install build-essential

# Export LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"

# uninstall and reinstall to match transformer version
RUN pip uninstall -y transformer_engine && pip install transformer_engine