# replicate-attention-sink
This repo is to replicate the paper [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) by Xiao et al. My goal is to replicate Table 1 from the paper. Specifically, I showed that adding four initial tokens as attention sinks drastically reduce the perplexity of `Llama-2-7b` model on a 60k-token long document. 
Additionally, the paper mentioned the importance of modifying positional embedding (`This method of assigning positional embedding within the cache is crucial to StreamingLLM’s functionality, ensuring that the model operates efficiently even beyond its pre-training attention window size.`) I demonstrated the importance of implementing positional embedding within cache by comparing results with and without changing the positional embedding.

The original paper experimented with cache size of `1024` with `Llama-2-13b` model. Due to memory constraint, I implemented with cache size of `512` with `Llama-2-7b` model. 

## Contributions
The [code](https://github.com/mit-han-lab/streaming-llm/blob/main/streaming_llm/pos_shift/modify_llama.py) that modifies positional embedding in the paper's original github repo is compatible with `transformers` version `4.33.1`. I modified the code (`modify_llama.py`) so it is compatible with current `transformers` version (`4.39.1`). The major differences between the two versions of `transformers` are that (1) `position_ids` changed from a Tensor object to a DynamicCache() instance and (2) The `LlamaRotaryEmbedding` class.

## Environment Setup
```sh
conda create -n attention-sink python=3.10
pip install -r requirements.txt
```
## Download data

The original paper experimented with the first document from [PG-19](https://github.com/google-deepmind/pg19) test set.
```
wget https://storage.googleapis.com/deepmind-gutenberg/test/10146.txt
```

## Experiments
1. Window attention (0 initial token):
```
python3 script.py --n-start 0 --output-dir outputs/window_attention
```
2. Use 4 initial tokens as attention sink without changing positional embedding:
```
python3 script.py --n-start 4 --output-dir outputs/sink-no-pos-shift
```
3. Use 4 initial tokens as attention sink with positional embedding based in cache position:
```
python3 script.py --n-start 4 --enable-pos-shift --output-dir outputs/sink-with-pos-shift
```


