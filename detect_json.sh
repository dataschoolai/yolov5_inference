input=/media/buntuml/DATASET/TEST_CASE/dam/wall/1
output=dataset_json
weights=efflorescence.pt
python detect_json.py --input $input --output $output --weights $weights

