import dill


with open("model_RandomForestClassifier_TfidfVectorizer.pkl", "rb") as fp:
    model = dill.load(fp)

print(model)
