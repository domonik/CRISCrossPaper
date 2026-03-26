


mkdir -p configs
python -m configGenerators.create_pretrain_array_ablation.py
python -m configGenerators.create_pretrain_array.py
python -m configGenerators.create_artificial_pretrain_params.py
python -m configGenerators.create_crisprat_ag_params.py
