import json
with open('training_data_1k.json', 'r') as f:
    data = json.load(f)
remapped = []
for t, g in data.items():
    for s, p in g.items():
        for i, q in p.items():
            if 'reviewed' not in q:
                q['reviewed'] = False
                q['topic'] = t
                q['skill'] = s
            remapped.append(q)
with open('remapped_training_data_1k.json', 'w') as f:
    json.dump(remapped, f)
