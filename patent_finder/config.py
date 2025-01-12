import argparse

class Config:
    def parse_args(self):
        parser = argparse.ArgumentParser(description="A simple argument parser")
        # General
        parser.add_argument('--name', default='none', type=str)
        parser.add_argument('--llm_name', default='gpt-4o', type=str)
        parser.add_argument('--max_fulltext_len', default=96000, type=int, help='maximum number of tokens in the full text, 24k tokens for GPT-4o by default')

        # Patent Files
        parser.add_argument('--cache_root', default='cache/google_patent', type=str)
        parser.add_argument('--patent_extract_method', default='google_patent', type=str)
        parser.add_argument('--image_parser_endpoint', default='', type=str)
        parser.add_argument('--max_page_num', default=300, type=int, help='maximum number of pdf pages. Use in the end2end baseline, 300 for gemini by default')

        # Test
        parser.add_argument('--test_method', default='random', type=str)
        parser.add_argument('--test_method_name', default='claim_check', type=str)
        parser.add_argument('--test_data_path', default=None, type=str)
        parser.add_argument('--test_result_path', default=None, type=str)
        parser.add_argument('--test_num_workers', default=2, type=int)
        parser.add_argument('--test_neg_aug', default=0, type=int)

        args = parser.parse_args()
        for key, value in vars(args).items():
            setattr(self, key, value)
        return args
config = Config()
config.parse_args()
print("=== Config ===")
for key, value in vars(config).items():
    print(f"{key}: {value}")
print("==============")