install:
	pip3 install -r requirements.txt

format:
	black -l 120 *.py

get-data:
	mkdir data
	get-pretrain-data
	get-train-data

get-pretrain-data:
	gsutil cp gs://fras/lakh_nes_pretrain/tx1_pretrain_final.zip data/
	unzip tx1_pretrain_final.zip

get-train-data:
	gsutil cp gs://fras/nes_tx1_full_seq_size_300.txt data/
	gsutil cp gs://fras/nes_tx1_vocab.txt data/
	gsutil cp gs://fras/nes_seq_size_300_nes_tx1_full_seq_size_300.txt data/

common-fixes:
	pip uninstall tensorflow
