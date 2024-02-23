# quantizeHFmodel
Accepts Hugging Face models, and automatically downloads, quantizes, and reuploads the model to an HF repo of your choice.

Remember to export your Hugging Face Token like so:
```
export HUGGING_FACE_HUB_TOKEN="YOUR_TOKEN"
```

An example of using the script is like:
```
python3 quantizeHFmodel.py fireworks-ai/firefunction-v1
```
