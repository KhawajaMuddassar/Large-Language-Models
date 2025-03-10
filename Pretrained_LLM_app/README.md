# ChatGPT like Interface for interacting with Pretrained LLM



To develop this user interface, we will utilize open-source resources [Chainlit Python package](https://github.com/Chainlit/chainlit).

&nbsp;
## Dependencies

## Install the `chainlit` package via

```bash
pip install chainlit
```
&nbsp;
## Run `app` code

1. [`app_openai.py`](app_openai.py): This file loads and uses the original GPT-2(small) weights from OpenAI. 
2. [`app_gptlocal.py`](app_gptlocal.py): This file loads and uses the GPT-2 weights we trained.


Run one of the following commands from the terminal to start the UI server:

```bash
chainlit run app_openai.py
```

or

```bash
chainlit run app_gptlocal.py
```

Running one of the commands above will typically launch a new browser tab to interact with the model.
If the tab doesn't open on its own, verify the terminal command and copy the local URL (usually http://localhost:8000) into your browser's address bar.
