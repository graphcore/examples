Copyright (c) 2023 Graphcore Ltd. All rights reserved.

# Guidelines for Writing a Notebook

These guidelines are intended to be used in conjunction with the
[notebook template](notebook_template.ipynb).

The aims of these guidelines are to help you write, edit and review a
demo notebook.

The aim of writing a notebook is to demonstrate how IPUs can be used to
solve specific problems. The expectation is that users will then use
IPUs to solve their problem.

## About these guidelines

These guidelines are separated into two parts:

-   [Part A](#part-a-notebook-structure) gives details about the structure of the notebook that is
    summarised in the [notebook template](notebook_template.ipynb).
    -   The template is very general and while there are "compulsory"
        sections, these are to ensure that all relevant information is
        included.
-   [Part B](#part-b-best-practice-guidelines) describes the best practice to help you create notebooks that
    are consistent with other Graphcore notebooks.

## Further resources

These guidelines summarise all the information you need to know to write
a notebook, but you can find more information in the following:

-   General Graphcore documentation guidelines for
    [writing](https://graphcore.atlassian.net/wiki/spaces/DOCS/pages/3019341893/Writing+guidelines)
    and
    [reviewing](https://graphcore.atlassian.net/wiki/spaces/DOCS/pages/3019374624/Reviewing+guidelines).
    These guidelines describe the common tips that apply to all
    technical content, for example we don't use latinisms like "etc."
    or "e.g." because not everyone knows exactly what these mean.
-   [Notebook
    guidelines](https://graphcore.atlassian.net/wiki/spaces/PM/pages/3094381247/Notebooks+guidelines):
    descriptions of the notebook naming convention and the high-level
    goals of notebooks.
-   [Writing a Paperspace
    notebook](https://graphcore.atlassian.net/wiki/spaces/PM/pages/3098345498/Writing+a+Paperspace+notebook):
    guidelines for writing notebooks for Paperspace, including
    requirements, dependencies and configuration.
-   [Common code interface to simplify use of existing applications in
    notebooks](https://graphcore.atlassian.net/wiki/spaces/PM/pages/3164995668/Making+applications+notebook+ready+RFC)
-   [Notebook user
    personas](https://graphcore.atlassian.net/wiki/spaces/PM/pages/3157131517/Notebook+personas#Ellie%3A-The-Data-Scientist%2C-Business-Analysis%2C-Consultant):
    lists of the characteristics of the different notebook user
    personas. These lists help you to define what to include in a
    notebook and how much detail to include so that the notebook is
    relevant for the targeted persona.
-   [Using Graphcore's Single source tool to write and manage
    notebooks](https://github.com/graphcore/single-source-tool-private)
-   [Tools for supporting Paperspace Gradient
    integration](https://github.com/graphcore/paperspace-automation)

## Setting up a notebook environment with IPU access

To set up a notebook environment with IPU access, you can use :

-   the
    [`local-gradient-notebook.sh`](https://github.com/graphcore/paperspace-automation/blob/main/local-gradient-notebook.sh)
    file in `paperspace-automation`.
-   the instructions in [Using IPUs from Jupyter
    Notebooks](https://github.com/graphcore/examples/tree/master/tutorials/tutorials/standard_tools/using_jupyter)
-   VS Code and [Using VS Code with the Poplar SDK and
    IPUs](https://github.com/graphcore/tutorials/tree/sdk-release-3.1/tutorials/standard_tools/using_vscode)
    to set it up to work with the Poplar SDK.

You can also use
(JupyterLab)[https://blog.jupyter.org/jupyterlab-is-ready-for-users-5a6f039b8906]
to develop the notebook. JupyterLab is an interactive development
environment for working with notebooks, code and data. Refer to the
[Installing Jupyter](https://jupyter.org/install#jupyterlab) page for
information on installing JupyterLab.

## Part A: Notebook structure

This part of the guidelines describes the different parts of the
[notebook template](notebook_template.ipynb):

-   [Copyright notice](#copyright-notice)
-   [Notebook title](#notebook-title)
-   [Introduction](#introduction)
-   [Buttons](#buttons)
-   [Environment setup](#environment-setup)
-   [Dependencies and configuration](#dependencies-and-configuration)
-   [Main body of notebook](#main-body-of-notebook)
-   [Conclusions and next steps](#conclusions-and-next-steps)

### Copyright notice

The top of each notebook should have the copyright notice. For more
information on what dates to use, refer to the description in the
[Copyright
dates](https://graphcore.atlassian.net/wiki/spaces/DOCS/pages/2876276824/Copyright+dates)
Confluence page.

### Notebook title

The notebook title must follow the following convention:

> [Solution/Task] on IPUs using [Model] - Inference/Fine-tuning
> [optional where applicable, for example "using your own dataset"]

This convention helps to make our notebooks more discoverable by search
engines (SEO).

Note: it is not mandatory to specify the framework in the notebook name
as we group the notebooks by framework. However, feel free to add the
framework if you feel it is important to highlight the framework, for
example PyTorch Geometric.

Some examples of notebook names that show how the naming convention is
used:

-   Text-to-Image Generation on IPUs using Stable Diffusion - Inference
-   Text Guided In-Painting on IPUs using Stable Diffusion - Inference
-   Multi-label Classification on IPUs using ViT - Fine-tuning
-   Image Classification on IPUs using ViT - Fine-tuning
-   Question-Answering on IPUs using BERT - Fine-tuning
-   Node Classification on IPUs using Cluster-GCN - Training with PyTorch Geometric

### Introduction

We aim to have our notebook app demos focused on the problem the user is
trying to solve. To help you do this correctly please read and
understand the [user
personas](https://graphcore.atlassian.net/wiki/spaces/PM/pages/3157131517/Notebook+personas#Ellie%3A-The-Data-Scientist%2C-Business-Analysis%2C-Consultant)
that our notebooks are targeted at. Use these personas to help you
decide what content to include and how much detail to include. When in
doubt ask yourself: "Will the targeted persona care about this?".

To support this, the first paragraph of text should contain all the key
information to help users rapidly decide if this notebook can help them
solve their problem. It takes time to run a notebook and we don't want
to waste a user's time. So, the intro paragraph must be very clear
about:

-   The task/business problem this demo presents a solution for
-   The features that are used in the demo, focussing on big-picture
    deep learning features, for example distributed training and not I/O
    overlap. Remember, we are trying to solve a user's problem, not
    sell an IPU-specific feature.

Our proposal is that each notebook start with the following three items:

-   a table summarising what we are going to do
-   a very short introduction covering the context that this notebook
    covers (3-5 sentences)
-   a bulleted list with clear "steps to resolution" stating what the
    user needs to do to solve their problem on the IPU
-   links to additional, related resources, including a button to join
    the Graphcore Slack community.

#### Summary table

Fill in this table as best as you can. Ask if you are unsure. Your
reviewers should be able to correct the information if there are any
mistakes.

  --------------------------------------------------------------------------------
  Domain    Tasks   Model   Datasets    Workflow      Number of IPUs Execution
                                                                     time
  --------- ------- ------- ----------- ------------- -------------- -------------
  NLP       Q&A     GPT-J   Glue-MNLI   Training,     recommended:   20Xmn
                                        evaluation,   16XX (min: 4X) (X1h20mn)
                                        inference

  --------------------------------------------------------------------------------

#### Intro paragraph

The introduction to the notebook should be about three to five sentences
long. It should establish the context for the notebook, focussing on the
problem the notebook solves. It should not mention any IPU-specific or
framework-specific features. The mindset is that anything that is
non-standard is a barrier to entry, and will risk the user giving up.

##### Example of a good intro paragraph

This notebook demonstrates speech transcription on the IPU using the
[Whisper implementation in the Hugging Face Transformers
library](https://huggingface.co/spaces/openai/whisper) alongside
[Optimum Graphcore](https://github.com/huggingface/optimum-graphcore).

> Whisper is a versatile speech recognition model that can transcribe
> speech as well as perform multi-lingual translation and recognition
> tasks. It was trained on diverse datasets to give human-level speech
> recognition performance without the need for fine-tuning.

#### Learning outcomes

Next, is a clear bullet list summarising the learning outcomes of the
demo. Each outcome should be of the form:

-   what the user will do (active verb) [and (optionally) how they do
    it]. Jargon, if any, goes to the end of the bullet point.

Examples of learning outcomes: In this demo, you will learn how to:

-   generate text in 5 lines of code with GPT-J on the Graphcore IPU
-   use GPT-J to answer questions and build prompts to reliably use
    text-generation for more specific NLP tasks
-   explore the sensitivity of the model to the prompt format and
    compare the model performance with 0-shot (single example) and
    few-shot prompting
-   improve text generation throughput using batched inference
-   understand the limitations of the base GPT-J checkpoint when it
    comes to more complex NLP tasks
-   use the model to identify whether statements agree or disagree
    (entailment). For this more complex task, we show the benefit of
    fine-tuning and load a checkpoint from the Hugging Face Hub which
    achieves much better performance on this specific task

#### Links to other resources

In the last paragraph, you should guide users to other resources, for
example other notebooks, documentation, and tutorials, to make it
possible for a user who realizes this is not the demo for them to find
the right content. Typically if this notebook is an inference demo, then
you can point to fine-tuning, and vice-versa. Point to a bigger/smaller
model to do something similar.

We include the link to join our Slack Community.

[![Join our Slack
Community](https://img.shields.io/badge/Slack-Join%20Graphcore's%20Community-blue?style=flat-square&logo=slack)](https://www.graphcore.ai/join-community)

### Buttons

We like to make it easy for users to run the notebook and to connect to
our Slack Community. So, we add the following buttons to the notebook.

[![Run on
Gradient](../../gradient-badge.svg)](https://console.paperspace.com/github/%3Cruntime-repo%3E?machine=Free-IPU-POD4&container=%3Cdockerhub-image%3E&file=%3Cpath-to-file-in-repo%3E)
[![Join our Slack
Community](https://img.shields.io/badge/Slack-Join%20Graphcore's%20Community-blue?style=flat-square&logo=slack)](https://www.graphcore.ai/join-community)

#### Link for the Run on Gradient button

Once the notebook is available on Paperspace Gradient we like to have a
"Run on Gradient" button. The link for the button needs to be
configured. The example above shows the convention for how to form the
link.

-   The SVG image file for the button should be a local file in the repo
    (as shown in the example above). You can also use the [image file on
    Paperspace](https://assets.paperspace.io/img/gradient-badge.svg) but
    this is not reliable as Github's caching occasionally breaks.
-   `<runtime-repo>` represents the "organisation/repository-name" of
    the public repository that contains the notebook.
-   `<dockerhub-image>` is the name and tag of a public Docker Hub
    container.
-   `<path-to-file-in-repo>` is the location of the notebook inside the
    repo starting with a leading `/`.

Note the part after the `?` in the link needs to be URL-encoded. You can
use an online [URL encoder](https://www.urlencoder.org/) or you can use
the [Paperspace link
builder](https://docs.paperspace.com/gradient/notebooks/run-on-gradient/).

Refer to the Confluence page `Generating a short URL for a Paperspace notebook <https://graphcore.atlassian.net/wiki/spaces/PM/pages/3219194169/Generating+a+short+URL+for+a+Paperspace+notebook>`__ for more information about managing these links and accessing the list of existing links.

Example of a fully-formed link for the "Run on Gradient" button:
<https://console.paperspace.com/github/gradient-ai/Graphcore-Pytorch?machine=Free-IPU-POD4&container=graphcore/pytorch-jupyter:3.1.0-ubuntu-20.04-20230104&file=/temporal-graph-networks/Train_TGN.ipynb>

```
    # Make imported python modules automatically reload when the files are changed
    # needs to be before the first import.
    %load_ext autoreload
    %autoreload 2
    # TODO: remove at the end of notebook development
```

### Environment setup

This section describes the system setup steps. In general, it is a
standard piece of text in a single Markdown block which can be easily
removed for the Paperspace version of the notebook. We propose the following standard text, that you can adapt:

> The best way to run this demo is on Paperspace Gradient's cloud IPUs
> because everything is already set up for you.

> [![Run on
> Gradient](../../gradient-badge.svg)](https://console.paperspace.com/github/%3Cruntime-repo%3E?machine=Free-IPU-POD4&container=%3Cdockerhub-image%3E&file=%3Cpath-to-file-in-repo%3E)

> To run the demo using other IPU hardware, you need to have the Poplar
> SDK enabled {and a PopTorch/TensorFlow wheel installed}. Refer to the
> [Getting Started
> guide](https://docs.graphcore.ai/en/latest/getting-started.html#getting-started)
> for your system for details on how to do this. Also refer to the
> [Jupyter Quick Start
> guide](https://docs.graphcore.ai/projects/jupyter-notebook-quick-start/en/latest/index.html)
> for how to set up Jupyter to be able to run this notebook on a remote
> IPU machine.

### Dependencies and configuration

We suggest putting the dependencies and configuration information in its
own section, mainly to allow for the easy removal of the [Environment
setup](#environment-setup) section for the Paperspace version of the
notebook.

#### Dependencies

You can install requirements directly from a notebook. You can:

1.  Run commands by starting the line in a code cell with `!`, as shown
    in the first code block below.
2.  Install Python requirements with the `%pip` [magic
    command](https://ipython.readthedocs.io/en/stable/interactive/magics.html).

Use these methods to make it easier for your user to set up the
environment they need.

```
    !apt update -y
    !apt install -y <reqs...

    # example running make:
    import os

    code_directory = os.getenv("OGB_SUBMISSION_CODE", ".")
    !cd {code_directory} && make -C data_utils/feature_generation
    !cd {code_directory} && make -C static_ops
```

```
    %pip install  -r requirements.txt
```

#### Configuration

In order to provide the best possible experience on Paperspace, you need
take into account the location of datasets, checkpoints, and Poplar
cached executables, as well as the number of available IPUs. A
description of why this is important is given in [Writing a Paperspace
notebook](https://graphcore.atlassian.net/wiki/spaces/PM/pages/3098345498/Writing+a+Paperspace+notebook).

To make it easier for you to run this demo, we read in some
configuration related to the environment you are running the notebook
in.

```
    import os

    number_of_ipus = int(os.getenv("NUM_AVAILABLE_IPU", 16))
    pod_type = os.getenv("GRAPHCORE_POD_TYPE", "pod16")
    executable_cache_dir = os.getenv("POPLAR_EXECUTABLE_CACHE_DIR", "/tmp/exe_cache/")
    dataset_directory = os.getenv("DATASETS_DIR")
    checkpoint_directory = os.getenv("CHECKPOINT_DIR")
```

As the notebook writer, you only need to define a variable if you intend
to use the value in the rest of the notebook. You can choose to use the
default values of any variables if it is better for your development
workflow. When you write the notebook, you need to use those variables
that you have defined to configure the execution.

Note on Poplar executables: We also set the standard PyTorch, PopART
and TensorFlow environment variables, so if you do not customise the
behaviour then you don't need to read them from the environment. For
more information refer to [Writing a Paperspace notebook](https://graphcore.atlassian.net/wiki/spaces/PM/pages/3098345498/Writing+a+Paperspace+notebook).

### Main body of notebook

The headings of your sub-sections for the main body of your notebook
should be related (preferably identical) to the list of learning
outcomes. This makes it easier for your user to match what is expected
from the learning outcomes with what is described in these detailed
sections.

These sub-sections contain the main content of the demo.

#### About notebook sections

Notebooks should have one top level section `# Notebook Title` all other
sections should use 2 or more #-characters to create section headers.
Don't hesitate to split your text into several Markdown blocks for
clarity when reviewing. These blocks will appear together in the
rendered output.

#### Sub-sub sections

Add additional #-characters to create nested sections and make your
tutorial easier to navigate.

### Conclusions and next steps

This section should describe the conclusions from this notebook:

-   Summarise the main steps that were performed in the demo making it
    clear what your user got to do. This can be similar to the learning
    outcomes listed at the beginning of the notebook, but can contain
    more details. Try to link the specific feature, method or class that
    was used to achieving a specific outcome. Remember we want to
    highlight how we can solve the user's problems not sell a feature.
    (short paragraph: 3-6 sentences)

-   Provide resources for the user's next steps. These can be links to
    other tutorials, to specific documentation (for example user guides,
    tech notes), to code examples in the public Graphcore
    [examples](https://github.com/graphcore/examples) repo, or to other
    deployments. (2-4 suggestions)

    If you want to link to a notebook in the same runtime, then point
    the user to the file is rather than using an explicit. For
    example: "Please see our [name of tutorial] tutorial in
    `<folder_name>/<notebook_name>.ipynb`. For relative links, the
    Paperspace platform will download the file locally if the machine
    is running and if the machine is not running it will throw a 404
    error. New windows are opened for full path links.

## Part B: Best practice guidelines

This section covers the best practice guidelines that will help you to write notebooks and contains the following information:

* [Working with GitHub repos](#working-with-the-github-repos)
* [How to get it done fast](#how-to-get-it-done-fast)
* [Suggestions for an engaging notebook](#suggestions-for-an-engaging-notebook)
* [Useful tips and known challenges](#useful-tips-and-known-challenges)
* [Licensing and copyright](#licensing-and-copyright)
* [Asking for help](#asking-for-help)

### Working with the GitHub repos

In general the process you will follow is as follows:

1.  Work on a fork of the upstream repo.
2.  Create your working branch.
3.  Write your notebook.
4.  Push to GitHub.
5.  Create a PR from the GitHub web interface.
6.  Add reviewers.
7.  Address review comments.
8.  After the PR is approved, merge (if you have permission) or send a
    link to the repo owner asking that the PR be merged.

### How to get it done fast

#### General guidelines

-   It works well to work in a team of two people (writer and reviewer).
    The writer drives the code development and writes the content. The
    reviewer helps with writing the content and reviews.
-   The writer and reviewer person should expect to each spend at least
    1 week on a notebook. This is 1 week of only working on a notebook.
-   The writer and reviewer should collaborate very closely. Feedback
    between them should range from a few minutes to a couple of hours,
    no more.

#### Writer guidelines

-   Make sure the model you have chosen is good enough to do what you
    need it to. Don't write the text until the code does what you need
    it to.

-   Before you write anything, "pitch" the story the notebook is going
    to tell with the code to:

    -   the reviewer
    -   the Product owner
    -   another engineer with relevant expertise

-   You own the code and any major structural changes in the notebook.

-   Follow the general Graphcore [writing
    guidelines](https://graphcore.atlassian.net/wiki/spaces/DOCS/pages/3019341893/Writing+guidelines#Formatting)
    for formatting, grammar and terminology.

#### Reviewer guidelines

-   You don't own any code changes or major restructuring of the
    notebook. This is owned by the writer.

-   Follow the general Graphcore [reviewing
    guidelines](https://graphcore.atlassian.net/wiki/spaces/DOCS/pages/3019374624/Reviewing+guidelines).
    You must also be familiar with the general Graphcore [writing
    guidelines](https://graphcore.atlassian.net/wiki/spaces/DOCS/pages/3019341893/Writing+guidelines#Formatting)
    for formatting, grammar and terminology.

-   When reviewing, you can:

    -   create your own PR
    -   raise a PR on top of the current PR
    -   press . in the PR which opens VS Code in your browser and you
        can make changes directly. Just remember to commit and push the
        changes when you are done.

### Suggestions for an engaging notebook

#### Focus on making the notebook interactive

-   Have clear user tunable parameters that are well-described.

-   Highlight the settings/parameters [the targeted
    user](https://graphcore.atlassian.net/wiki/spaces/PM/pages/3157131517/Notebook+personas)
    cares about.

-   Hide things the user does not care about (for example IPU
    specificity, framework). You can do this by importing a `utils`
    file.

-   Add notebook widgets if appropriate.

-   Show (don't tell) by printing a variable. If the variable is not
    human readable then overwrite the `repr` function to give a more
    useful description.

#### Enable the user to do what they want

-   To make the code in applications easy to use, implement the
    interface in `utils/templates/api.py`. Refer to [Applications common
    code
    interface](https://graphcore.atlassian.net/wiki/spaces/PM/pages/3164995668/Making+applications+notebook+ready+RFC#Proposal)
    for more information. Write the notebook in such a way that they
    user can easily:

-   Change the model checkpoint

-   Change the dataset. Make the format of the data obvious.

-   Save a checkpoint

-   Evaluate the model to:
    1. get a feel for it
    2. run an evaluation

#### Notebooks that demonstrate the good practice described in this template

    This template is intended as a helper to notebook developers, but
    flexibility is allowed. So, the following are examples of
    notebooks that you can use for inspiration. Note they may not
    follow the template exactly, but use what is suitable for the
    task.

-   [Text Entailment on IPU using GPT-J - Fine-tuning](../../nlp/gpt_j/popxl/finetuning.ipynb)

-   [Text generation with GPT-J
    6B](../../nlp/gpt_j/popxl/GPTJ-generative-inference.ipynb)

-   [Fast sentiment analysis using pre-trained models on Graphcore
    IPU](https://github.com/graphcore/Graphcore-HuggingFace-fork/blob/main/natural-language-processing/sentiment_analysis.ipynb)

-   [Real Time Name Entity Recognition on the
    IPU](https://github.com/graphcore/Graphcore-HuggingFace-fork/blob/main/natural-language-processing/name-entity-extraction.ipynb)

-   [Stable Diffusion Image-to-Image Generation on
    IPU](https://github.com/graphcore/Graphcore-HuggingFace-fork/blob/main/stable-diffusion/image_to_image.ipynb)

### Useful tips and known challenges

#### Working with `argparse` and command line arguments

If you have encountered problems related to `argparse` while writing a
notebook, these tips may help you resolve your problem:

-   Try to disentangle your application from any argument parsing logic.
-   Manually create an
    [`argparse.Namespace`](https://docs.python.org/3/library/argparse.html#argparse.Namespace).
-   Define custom parsing logic in your app to detect when its running
    in a Jupyter Notebook, for example as shown in the [simple parsing
    utilities](https://github.com/graphcore/examples-utils/blob/f8673d362fdc7dc77e1fee5f77cbcd81dd9e4a2e/examples_utils/parsing/simple_parsing_tools.py#L118).

Often with these kinds of problems, the issue is rooted in the structure
of the app, so consider using the [Applications common code
interface](https://graphcore.atlassian.net/wiki/spaces/PM/pages/3164995668/Making+applications+notebook+ready+RFC#Proposal)
to write an app that is easier to use.

#### Detaching from IPUs

Notebooks continue running after the last cell has been run, so you
need to make sure that all IPUs are released at the end. This ensures
that other users have resources available to run their notebooks.

```
    # For PopTorch
    if model.isAttachedToDevice():
    model.detachFromDevice()

    #For TensorFlow2
    from tensorflow.python import ipu
    ipu.config.reset_ipu_configuration()
```

### Licensing and copyright
Pay attention to any content you source from outside Graphcore. These can be code, images, or extracts of research papers.

Check with the legal team (legal@graphcore.ai)) whether it is ok to use the content you want to source from outside Graphcore *before* starting to work with it. This can avoid you wasting time and effort.

If the content can be used, you must add clear and precise attributions for that content by describing the license that applies, and state what you have changed (if anything).

For example, you may add the following to an extract of a website:

> Extract of the MMLU leaderboard from [Papers With Code](https://paperswithcode.com/sota/multi-task-language-understanding-on-mmlu) (CoT = Chain of Thought). This content is licensed under the terms of the [CC BY-SA](https://creativecommons.org/licenses/by-sa/4.0/) license. No changes were made.

For an image, you may add something like:

> This application uses images from [name of source](link to source) which are made available under the terms of the [name of license].

### Asking for help

If you come across a situation that is not described in these
guidelines, then post the question on the
[#internal-paperspace-graphcore](https://graphcore.slack.com/archives/C02969A3Z52)
Slack channel.
