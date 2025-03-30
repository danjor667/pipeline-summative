FROM ubuntu:latest
LABEL authors="dan"

ENTRYPOINT ["top", "-b"]