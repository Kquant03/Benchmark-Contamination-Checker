# A Quick Way To Check For Contamination In Datasets

This is just a simple script to use roberta to find paraphrased or directly copied benchmark examples in a dataset.

You can replace the files with the ones you intend to use and set the batch size and similarity threshold to whatever you want.

It's made to run on GPU because it's faster to do it this way.

```
git clone https://github.com/Kquant03/Benchmark-Contamination-Checker
cd Benchmark-Contamination-Checker
pip install -r requirements.txt
python3 cosine.py
```

I tested it against benchmark questions I paraphrased myself and it worked fine. I needed this for a dataset I generated with Nemotron.

Note: If you're going to test it against another benchmark, reformat it to where the only felds are "question" and "answer"

Also, it's made to test against ShareGPT datasets. So my benchmark is reformatted to question/answer jsonl and my dataset I'm testing is ShareGPT.

Finally, I tested it on Runpod with an SXM A100 and it wasn't much faster than my 3060 even with 1k+ batch size.
