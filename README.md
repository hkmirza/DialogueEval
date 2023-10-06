# Usage
1. Run ```python start_server.py```
2. Open `mturk_free_topic.html` or  `mturk_ice_breaker.html` in Chrome.
3. Now you can start chat with bots.

# Requirements

- python=3.8
- transformers
- pytorch
- tqdm
- flask
- flask-cors

# Files

## MTurk user interface files

The contents of these two files can be  directly copied and pasted to MTurk.
- `mturk_free_topic.html`: HMTL code of free topic (please check the comments at line 5 and line 397)
- `mturk_ice_breaker.html`: HMTL code of ice-breaker (please check the comments at line 5 and line 400)

## Server
- `start_server.py`: it runs a Python server. 
- `utils.py`: it is the code of dialogue models.

## Other files
- `degraded_random_responses_filtered.txt`: response candidates of the qc model.
- `personas.txt`: persona candidates
