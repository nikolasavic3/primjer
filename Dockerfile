FROM mambaorg/micromamba:1.5-jammy

# Copy environment file
COPY --chown=$MAMBA_USER:$MAMBA_USER environment-docker.yml /tmp/env.yml

# Create environment
RUN micromamba create -f /tmp/env.yml -y

# Set up working directory
WORKDIR /app
COPY --chown=$MAMBA_USER:$MAMBA_USER . /app

# Activate environment and run
CMD ["micromamba", "run", "-n", "primjer-docker", "python", "reproducibility_test.py"]
