# Hypothesis Forge

Hypothesis Forge is a local web app for exploring scientific ideas. You upload research material like PDFs, CSV files, images, and notes. The app then helps you:

- summarize evidence
- spot missing information
- generate testable hypotheses
- critique those hypotheses
- suggest simple experiment plans

This project is meant for demos, class projects, and early-stage brainstorming. It is not a tool for proving scientific truth.

## What The App Does

After you upload files, the app can:

- read text from PDFs
- optionally search external scientific literature on the web
- inspect CSV datasets and summarize patterns
- look at uploaded images
- combine everything into possible knowledge gaps
- generate ranked hypotheses
- create experiment plans for the strongest hypotheses
- save project progress locally and reopen earlier projects later
- export results as JSON, Markdown, or CSV

## Good News: You Can Run It Without API Keys

The app has a built-in mock/demo mode.

That means:

- you do not need NVIDIA credentials just to try it
- the app will still run from start to finish
- local file analysis still works
- the AI-generated parts use safe placeholder outputs when live models are not configured

This is the easiest way to demo the app.

## Before You Start

You need:

- Python 3.11 or newer installed on your computer
- a terminal or command prompt
- this project folder on your machine

If you do not have a coding background, that is okay. The steps below are written to be followed one at a time.

## Step-By-Step Setup

### 1. Open the project folder

Open a terminal in this folder:

`C:\Users\ashra\OneDrive\Desktop\BNL\Coding\AI-Jam`

If you are using VS Code, you can open the folder and then open the built-in terminal.

### 2. Create a virtual environment

This gives the project its own private Python space.

On Windows:

```bash
python -m venv .venv
```

### 3. Turn the virtual environment on

On Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

If it worked, you should see something like `(.venv)` at the start of the terminal line.

### 4. Install the required packages

```bash
pip install -r requirements.txt
```

This may take a minute or two.

### 5. Create your local settings file

Make a copy of `.env.example` and name the copy `.env`.

If you prefer using the terminal, run:

```bash
Copy-Item .env.example .env
```

### 6. Decide whether to use demo mode or NVIDIA mode

If you just want the app to run:

- leave `.env` mostly empty
- the app will use mock/demo mode automatically

If you want live NVIDIA model calls:

- put your NVIDIA API key in `.env`
- add the model names you want to use

The app automatically reads `.env` when it starts.

## Example `.env` File

```env
NVIDIA_API_KEY=
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_CHAT_MODEL=
NVIDIA_EMBED_MODEL=
NVIDIA_VISION_MODEL=
NVIDIA_RERANK_MODEL=
```

## How NVIDIA Mode Works

If these are missing:

- `NVIDIA_API_KEY`
- `NVIDIA_CHAT_MODEL`

the app stays in mock/demo mode.

If you also provide optional model names, the app can additionally use:

- embeddings for retrieval
- vision analysis for images
- reranking support

Keep your API key private. Do not commit `.env` to git.

## Run The App

Start the app with:

```bash
streamlit run app.py
```

After that, Streamlit should open a browser tab automatically. If it does not, copy the local URL shown in the terminal into your browser.

## Easiest Demo Workflow

If this is your first time using the app, do this:

1. Start the app.
2. Type a project title.
3. Type a research question.
4. Upload one PDF.
5. Upload one CSV file.
6. Optionally upload one image and some notes.
7. Click `Run Co-Scientist Pipeline`.
8. Read the tabs from left to right:
   Evidence Summary, Data Analysis, Knowledge Gaps, Hypotheses, Experiment Plans, Final Report.
9. Download the outputs if you want to save the results.
10. Use the `Save Current Project` button if you want to come back to the same work later.

## External Literature Search

The app can optionally search external scientific papers using the research question you enter.

Important notes:

- this needs an internet connection
- search results are treated as extra literature inputs
- the app uses search-result abstracts, not full papers
- uploaded PDFs are still useful for deeper evidence extraction

## Saved Projects

The app can save your progress locally.

When you save a project, it stores:

- your project title and research question
- your notes and processing settings
- your latest generated results
- copies of uploaded files so the project can be reopened later

You can reopen old work from the `Saved Projects` section in the sidebar.

## What Each Input Type Does

- `PDF`: used for literature or paper text extraction
- `CSV`: used for basic dataset analysis
- `Image`: used for visual observations
- `Notes`: used as extra project context

The app is still useful even if you only upload PDFs and one CSV.

## Testing The Project

If you want to confirm the code is working, run:

```bash
python -m pytest
python -m compileall .
```

These checks make sure the core Python files and tests are in good shape.

## Project Structure

```text
hypothesis-forge/
  app.py
  README.md
  requirements.txt
  .env.example
  src/
    config.py
    nvidia_client.py
    schemas.py
    pdf_processor.py
    data_processor.py
    image_processor.py
    retrieval.py
    agents.py
    pipeline.py
    scoring.py
    export.py
    utils.py
  tests/
    test_schemas.py
    test_data_processor.py
    test_scoring.py
  sample_data/
    README.md
```

## Limitations

- The app generates ideas, not verified scientific conclusions.
- PDF extraction depends on whether the PDF contains readable text.
- Dataset analysis is exploratory and should not be treated as proof of causation.
- Live model outputs still need human review.
- Image analysis is limited when no vision model is configured.

## Safety Note

Use this app for brainstorming and early research planning only. Important claims should always be checked by a human, ideally with domain experts, primary sources, and proper scientific validation.
