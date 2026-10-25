import json

for name in ["eval", "test"]:
    data = json.load(open(f"outputs/detections_{name}.json"))
    boxes = sum(len(img["detections"]) for img in data)
    print(name, "images:", len(data), "detections:", boxes)
