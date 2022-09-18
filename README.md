# cog-prompt-parrot
### Quickstart
```
1. sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
2. sudo chmod +x /usr/local/bin/cog
3. Create a /model directory to store pytorch_model.bin and config.json
4. cog predict -i prompt="banana on a plate"
```

### Use Docker
```
cog build -t cog-prompt-parrot
docker run -d -p 11111:5000 --restart=always --gpus all cog-prompt-parrot
```
```
curl http://localhost:11111/predictions -X POST -H "Content-Type: application/json" \
  -d '{"input": {
    "prompt": "...",
}}'
```

