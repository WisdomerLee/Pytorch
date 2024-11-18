from tranformers import pipeline
import pandas as pd


task = "zero-shot-classification"
model = "facebook/bart-large-mnli"
classifier = pipeline(task=task, model=model)


documents = ["~~소설 혹은 논문의 내용~~"]

candidate_labels = ["crime", "fantasy", "history"]

res = classifier(documents, candidate_labels=candidate_labels)
res

pd.DataFrame(res[1]).plot.bar(x='labels', y='scores', rot=0, title='')

# multiple label의 경우는 아래와 같이
classifier(documents, candidate_labels=candidate_labels, multi_class=True)
