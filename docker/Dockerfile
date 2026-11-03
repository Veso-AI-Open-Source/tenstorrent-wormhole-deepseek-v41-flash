# Build context: a tt-metal checkout at tag v0.81.0-dev20261020, after `docker buildx bake -f dockerfile/docker-bake.hcl basic-dev` in that checkout.
#   docker buildx build -f docker/Dockerfile -t dsv41-tt:v081 --load "$TT_METAL_TREE"
FROM tt-metalium-basic-dev:local
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
COPY install_dependencies.sh /opt/tt_metal_infra/scripts/docker/install_dependencies.sh
COPY tt_metal/sfpi-info.sh /opt/tt_metal_infra/scripts/docker/sfpi-info.sh
COPY tt_metal/sfpi-version /opt/tt_metal_infra/scripts/docker/sfpi-version
RUN /bin/bash /opt/tt_metal_infra/scripts/docker/install_dependencies.sh --docker
RUN apt-get update && apt-get install -y --no-install-recommends git git-lfs libnuma-dev curl jq file \
    && apt-get clean && rm -rf /var/lib/apt/lists/* && git config --global --add safe.directory '*'
