FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04
MAINTAINER Luisa Zintgraf

# Ubuntu Packages
RUN apt-get update -y && apt-get upgrade -y && apt-get install -y libgl1-mesa-glx vim build-essential git wget libpng-dev

# Python (Miniconda)
RUN wget --quiet http://repo.continuum.io/miniconda/Miniconda3-4.3.31-Linux-x86_64.sh -O miniconda.sh
RUN bash miniconda.sh -b -p /opt/conda && rm miniconda.sh
ENV PATH /opt/conda/bin:$PATH

# Python Packages
RUN conda install numpy scipy scikit-learn scikit-image pandas flask -y
RUN conda install pytorch torchvision -c pytorch -y
RUN conda clean -y -a
RUN pip install spm1d pulp

# Section to get permissions right, and avoid running inside as root {{

    # Create a user matching the UID, and create/chmod home dir (== project directory)
    # (uid corresponds to breord in CS network)
    RUN useradd -d /project -u 12705 --create-home user
    USER user
    WORKDIR /project/
    ADD . /project/

# }}
