#!/bin/bash

python test_pipeline.py --name "gpt-o1" --test_method 'pipeline' --test_method_name "claim_check" --llm_name "gpt-o1" --test_num_workers 2 --test_data_path data/benchmark_v3.5-240-Oct28.json --test_result_path results/$name --max_fulltext_len 24576;