# Introduction to CI/CD with GitHub Actions

What is _GitHub Actions_?

> GitHub Actions is a versatile CI/CD platform enabling automation of build, test, and deployment pipelines, supporting workflows for pull requests and production deployments. It extends beyond DevOps, allowing triggering of workflows on various repository events, such as adding labels to new issues. With options to utilize GitHub's virtual machines or self-hosted runners, it offers flexibility for execution in Linux, Windows, or macOS environments, whether on GitHub's infrastructure or your own data center/cloud setup.

In this part, you will learn how to add GitHub Actions to a GitHub repository and add continuous testing to your codebase.

## The components of GitHub Actions

A GitHub Actions _workflow_ is triggered when an _event_ occurs in your repository. Examples of such events are _commits_, _pull requests_, or a new issue being opened.

A workflow consists of one or more _jobs_ which can run sequentially or in parallel. Jobs run inside their own _runner_ and have one or more _steps_ that either run a script or an _action_. Actions are reusable extensions that you can include in your workflow - think of them as (third-party) packages for your workflow.

![GitHub Actions Components demo](imgs/github_actions_components.png)
(Image taken from GitHub Actions documentation)

### Your first workflow

Workflows are configurable automated processed that run one or more jobs. They are defined in YAML files, checked into a GitHub repository in the `.github/workflows` directory.

A workflow must contain the following basic components:

1. One or more _events_ that will trigger the workflow. There are [_many_](https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows) events that can trigger workflows.
2. One or more _jobs_, each of which will execute on a _runner_ machine and run a series of one or more _steps_.
3. Each step can either run a script that you define or run an _action_, which is a reusable extension that can simplify your workflow. Third-party actions can be found on the [GitHub Marketplace](https://github.com/marketplace?type=actions).

Let's look at an example of a workflow file.

```yaml
# This is the name of the workflow. It will appear in the GitHub UI.
name: CI/CD with GitHub Actions

# `on` specifies the triggers of your workflow. This workflow is triggered by:
on:
# `push` events on the main branch
  push:
    branches:
      - main
# `pull_request` events on the main branch
  pull_request:
    branches:
      - main

# `jobs` groups together all jobs in this workflow
jobs:

# This job is named "build".
  build:
  # Configures the job to run on the latest version of an Ubuntu Linux runner. When used on github.com, this means that the job will execute on a fresh virtual machine hosted by GitHub.
    runs-on: ubuntu-latest

  # Groups together all steps in the "build" job.
    steps:

    # The first step is named "Checkout code".
      - name: Checkout code
    # The uses keyword specifies that this step will run v4 of the actions/checkout action. 
    # This is an action that checks out your repository onto the runner, allowing you to run scripts or other actions against your code (such as build and test tools). 
    # You should use the checkout action any time your workflow will use the repository's code.
        uses: actions/checkout@v4

    # Another step, this one builds and tests some JavaScript code.
      - name: Build and test
        run: |
          npm install
          npm run build
          npm run test

# Another job, this one is deploys the code built in the previous job.
  deploy:
  # The needs keyword specifies dependencies.
    needs: build
    runs-on: ubuntu-latest

    steps:
      - name: Deploy to production
        uses: some-action/deploy@v1
    # The with keyword is used to define arguments for an action.
        with:
          environment: production
          token: ${{ secrets.PRODUCTION_TOKEN }}

```

Now that you've seen what a workflow looks like, it is time to build your own. In the next section, we're going to guide your through setting up GitHub Actions on the ZHAW GitHub. Afterwards, it's your turn!

## Setup

Unfortunately, GitHub Actions are disabled on the ZHAW GitHub instance. For simplicity, we are going to use `act`, a local runner / emulator for GitHub actions, but you are of course free to use [github.com](https://github.com/) if you want to.
`act` supports Windows, macOS, and Linux. To install `act`, follow the instructions [here](https://nektosact.com/installation/index.html). `act` requires Docker (or any other container engine) - don't worry, the installation instructions also cover this part and you won't have to interact with docker beyond installing it.

The first time you run `act`, you will be prompted to select a container size. Select `Medium`.
If you are running on macOS with an ARM processor (M1, M2, M3, maybe more by the time you are reading this), you will have to add the flag `--container-architecture linux/amd64` to make the docker happy.

### What does `act` do?

In their own words,

> when you run `act` it reads in your GitHub Actions from `.github/workflows/` and determines the set of actions that need to be run. It uses the Docker API to either pull or build the necessary images, as defined in your workflow files and finally determines the execution path based on the dependencies that were defined. Once it has the execution path, it then uses the Docker API to run containers for each action based on the images prepared earlier. The environment variables and filesystem are all configured to match what GitHub provides.

## A workflow for python

Below you see a simple workflow for python, taken from the [official documentation page](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python#testing-your-code):

```yaml
name: Python package

on: [push]

jobs:
  build:

    runs-on: ubuntu-latest

    steps:
        # This action checks out your code.
      - uses: actions/checkout@v4
        # --- after this step, your code is available to the runner ---
        # This action sets up python - that's why it's called "Set up Python".
      - name: Set up Python
        # This is the version of the action for setting up Python, not the Python version.
        uses: actions/setup-python@v4
        with:
          # Semantic version range syntax or exact version of a Python version
          python-version: '3.x'
          # Optional - x64 or x86 architecture, defaults to x64
          architecture: 'x64'
        # --- after this step, python is available ---
      # Now you can do whatever you want with python:
      - name: Display Python version
        run: python -c "import sys; print(sys.version)"
```

You can find this workflow in `lab04/github_actions_intro/workflows/python_demo.yaml`.
We can run it with `act` by issuing the following command in the `lab04` directory:

```shell
./bin/act push -W github_actions_intro/workflows
```

`act push` simulates a push to your github repository.

### Your turn: Install dependencies

Copy or modify `lab04/.github/github_actions_intro/workflows/python_demo.yaml`. Add a step that updates `pip`.

<details>
  <summary>Hint</summary>

Add a step that executes the following command:

```shell
python -m pip install --upgrade pip
```

</details>

Next, install `lab04/github_actions_intro/requirements.txt`.
To add a new command, you can either repeat the `run` key

```yaml
steps:
- uses: actions/checkout@v4
- uses: actions/setup-python@v4
  with:
    python-version: '3.11'
- run: do something
- run: do something else
```

or use [multiline YAML strings](https://yaml-multiline.info/). For instance:

```yaml
steps:
- uses: actions/checkout@v4
- uses: actions/setup-python@v4
  with:
    python-version: '3.11'
- run: | 
    do something
    do something else
```
