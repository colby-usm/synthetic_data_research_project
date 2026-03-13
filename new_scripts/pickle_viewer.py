import pickle
from pprint import pprint
with open("refcoco/refs(unc).p", "rb") as f:
    data = pickle.load(f)
    print(len(data))
    print(type(data[0]))


    for i, item in enumerate(data):
        if item.get('category_id', 1000000) > 90:
            print(item)
        elif i % 10000 == 0:
            print(f"element {i}: {item}")

    #print("first 10")
    #for item in data[:10]:
    #    pprint(item)
    #    print("\n---\n")

    #print("last 10")
    #for item in data[-10:]:
    #    pprint(item)
    #    print("\n---\n")
