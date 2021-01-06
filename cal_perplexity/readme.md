# Calculate Perplexity

using GPT-2 to calculate the perplexity,  Install `pytorch_pretrained_bert`

```python
pip install pytorch_pretrained_bert
```

load model and tokenizer and then calculate perplexities:

```python
model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
model.eval()
# Load pre-trained model tokenizer (vocabulary)
tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

# calculate perplexity score on your sentence(string)
s = score(sentence)

```



# Plot 

You need to produce a 1-dimension array full of perplexities first, you can save the numpy data in a `npy` or `npz` file for convenient plot.

```python
# non_ppl, beam_ppl, normal_ppl means perplexity on different dataset, in 1-d numpy type.
# remove nan to get mean and medium 
non_ppl, beam_ppl, normal_ppl = remove_nan(i) for i in (non_ppl, beam_ppl, normal_ppl)

data = [normal_ppl, non_ppl, beam_ppl]
data = [sorted(i) for i in data]  
data = [np.log(sorted(i)) for i in data]  # calculate log(ppl) since ppl have a large span.

# There may be some very big ppl interference drawing, you can choose whether to remove them.
if clip: 
    qt1, medians, qt3 = get_percentile(data, [25, 50, 75])
    whiskers = np.array([adjacent_values(sorted_array, q1, q3) for sorted_array, q1, q3 in zip(data, qt1, qt3)])
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]
    data = [np.clip(_data, _min, _max) for _data, _min, _max in zip(data, whiskers_min, whiskers_max)]

# start to plot the figure    
fig, ax = plt.subplots(figsize=(5, 4))
ax.set_title('Perplexities')
ax.set_ylabel('rate')
ax.set_xlabel('log(ppl)')

# change color and label as you like 
ax.hist(data[0],  bins=30, density=True, histtype='bar', stacked=True, alpha=0.3, lw=3, label='normal', color='b')
ax.hist(data[1], bins=30, density=True, histtype='bar', stacked=True, alpha=0.3, lw=3, label='greedy', color='r')
ax.hist(data[2], bins=30, density=True, histtype='bar', stacked=True, alpha=0.3, lw=3, label='beam-search', color='g')

plt.legend()
# ax.hist(data, bins=20, density=True, histtype='bar', stacked=True)

plt.savefig('ppl.png')


```

