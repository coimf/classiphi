import json

with open("training_data_1k.json", 'r') as file:
    data = json.load(file)
rows = []
count = 0
for topic in data["topics"]:
    for skill in data["topics"][topic]:
        for problem in data["topics"][topic][skill]:
            entry = data["topics"][topic][skill][problem]
            if not entry['problem'] == "" and not entry['solution'] == "":
                count += 1
                rows.append({'problem':entry['problem'].replace('$', ''), 'topic':topic, 'skill':skill})
print(count)
