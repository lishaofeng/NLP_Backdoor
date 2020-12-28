# Usage

主要的函数是`greedy_generate` 和 `beam_generate`， 返回 `list` 格式的句子。

```python
# load word2vec dictionary
corpus_path = './data/SQuAD.npz'
d = np.load(corpus_path, allow_pickle=True)
_, word2ix, ix2word = d['data'], d['word2ix'].item(), d['ix2word'].item()

# initialize finetuned network
net = Net(len(word2ix), 128, 256)  # TODO: Why 128 and 256?
net = nn.DataParallel(net)

checkpoint = './checkpoints/SQuAD_19.pth'
cp = torch.load(checkpoint)
net.load_state_dict(cp.state_dict())


context = "On May 21, 2013, NFL owners at their spring meet-ings in Boston voted and awarded the game toLevi’s Stadium. An apple a day keeps the doctoraway. The $1.2 billion stadium opened in 2014. Itis the first Super Bowl held in the San FranciscoBay Area since Super Bowl XIX in 1985, and thefirst in California since Super Bowl XXXVII tookplace in San Diego in 200."

# 处理数据
sents = nltk.sent_tokenize(context)
context = process(clean_df(sents))
context = " ".join(context)

# greedy generete: return a list of words
results = greedy_generate(net, ix2word, word2ix, prefix_word=context)
sent = " ".join(poem).replace(' eop', '.')

# beam search generate: return a list of words
results = beam_generate(net, ix2word, word2ix, prefix_word=context)
sent = " ".join(poem).replace(' eop', '.')
```

**注意：数据集最好将最后一个context去掉，或者将数据集逆序之后再生成，否则容易在最后一步报错。**



## 一些设置的解释

### greedy generate

在`greedy_generate` 的 53-58行中，注释掉的地方可以控制句子的开头单词 `w` 从 list `Qs` 中选择，即生成问句。

```python
output, hidden = net(input, hidden)
input = input.data.new([word2ix.get(SOS, 1)]).view(1, 1)
# output, hidden = net(input, hidden)
# w = choice(Qs)
# results.append(w)
# input = input.data.new([word2ix[w]]).view(1, 1)
```



在 70-82 行中，控制当句子的单词长度小于5时，句子将不能结束；如果概率最高的词为 EOS，则选择概率次高的词

```python
 # constraints: not too short
    if i <= 5 and w in [EOS, '.']:
        count = 1
        while w in [EOS]:
            count += 1
            top_index = output_data.topk(count)[1][-1].item()
            if top_index == 0:
                count += 1
                top_index = output_data.topk(count)[1][-1].item()
                w = ix2word[top_index]
                input = input.data.new([top_index]).view(1, 1)
                results.append(w)
                continue
```



### beam generate

在 125-134 行中，127-128被注释掉的两行则是选择 list `Qs` 中的单词作为句子的开头，即生成问句。 

```python
output, hidden = net(input, hidden)
input = input.data.new([word2ix.get(SOS, 1)]).view(1, 1)
# output, hidden = net(input, hidden)
# input = input.data.new([word2ix[choice(Qs)]]).view(1, 1)
node = BeamNode(hidden, None, input.item(), 0, 1)  # hidden, prev, idx, logp, length

# start the queue
nodes = PriorityQueue()
nodes.put((-node.eval(), node))
qsize = 1
```

在 139 行中，`if qsize > 2000: break` 则限制了句子的最大长度。一般而言，`qsize` 越大，句子所被允许的最大长度越大。

在 145-147 行中，`qsize >= 200` 则限制了句子的最小长度。 `qsize` 的最小长度越大，则句子的最短长度越大。

```python 
if n.idx == word2ix.get(EOS, 1) and n.prev and qsize >= 200:
    endnode = (score, n)
    break
```

