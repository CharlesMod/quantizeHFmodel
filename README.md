# quantizeHFmodel
Accepts Hugging Face models, and automatically downloads and quantizes it. 

Generates q4_k_m","q5_k_m", "q8_0" by default.

Remember to export your Hugging Face Token like so:
```
export HUGGING_FACE_HUB_TOKEN="YOUR_TOKEN"
```

An example of using the script is like:
```
python3 quantizeHFmodel.py fireworks-ai/firefunction-v1
```
