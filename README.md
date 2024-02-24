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

I'm also hosting quantizeHQQ here - it does the same thing except quantizes with HQQ, theoretically yielding a better quality quant. However, this takes crazy amounts of VRAM to do, on the order of >100GB. I don't have that, but if this is useful to you, more power to you.  
