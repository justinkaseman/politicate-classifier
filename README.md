# politicate-classifier

API for Politicate's classification model

## Getting Started

To run this project for local development there are a few options:

1. Docker

Install and run Docker Desktop (https://docs.docker.com/docker-for-mac/install/)

Run the following commands in your terminal:

```
docker build .  -t politicate-classifier
docker run politicate-classifier

```

...but it is a bit annoying to re-build the Docker image after every local change, so -

2. Pipenv (or other python shell)

Ensure that Homebrew is installed by running:

```
brew -v
```

Install pipenv (https://pipenv-fork.readthedocs.io/en/latest/) by running:

```
brew install pipenv
```

To ensure hot reloading is on make sure you create an `.env` file in the root of this repository with the following:

```
FLASK_ENV=development
```

Install the project's dependencies, start the shell, and start flask by running:

```
pipenv install
pipenv shell
python api.py
```

In both case 1 and 2 you will start the API server on on http://localhost:5000/

## Common commands

To send a test request to your locally running API:

```
curl http://localhost:5000/score --request POST --header "Content-Type: application/json" --data '{"input": "Fill me in"}'
```
