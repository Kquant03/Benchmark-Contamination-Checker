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
