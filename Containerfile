FROM continuumio/miniconda3:latest
WORKDIR /elv

RUN conda create -n mlpod python=3.10.14 -y

RUN apt-get update && apt-get install -y build-essential && apt-get install -y libgl1 && apt-get install -y ffmpeg

RUN /opt/conda/envs/mlpod/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Create the SSH directory and set correct permissions
RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh

# Add GitHub to known_hosts to bypass host verification
RUN ssh-keyscan -t rsa github.com >> /root/.ssh/known_hosts

COPY setup.py .
RUN mkdir -p src

RUN /opt/conda/envs/mlpod/bin/pip install .

COPY src ./src
COPY config.yml run.py config.py .

ENTRYPOINT ["/opt/conda/envs/mlpod/bin/python", "run.py"]