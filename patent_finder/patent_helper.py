import os
import json
from utils.google_patent_lookup import google_patent_scrap

class PatentHelper():
    def __init__(
            self,
            cache_root='cache/google_patent',
            patent_extract_method='google_patent',
            image_parser_endpoint='',
        ):
        self.cache_root = cache_root
        self.patent_extract_method = patent_extract_method
        self.image_parser_endpoint = image_parser_endpoint
    
    def replace_pdf_image(self, text, image_list):
        for image_dict in image_list:
            image_url = image_dict["link"]
            if 'caption' not in image_dict:
                # Skip empty images
                continue
            if image_url not in text:
                continue
            
            mol_string = image_dict["caption"]
            if not image_dict["markush"]:
                mol_string = image_dict["caption"].replace('<sep>', '')
            image_text = '\\begin{molecule}\n\caption{MOL_STRING}\n\end{molecule}'.replace('MOL_STRING', mol_string)
            text = text.replace(image_url, image_text)
        return text
    
    def google_patent_extract(self, patent_id):
        """
        get text and markush claims from the pdf
        """
        cache_path = f'{self.cache_root}/{patent_id}'
        patent_info = google_patent_scrap(cache_path, patent_id, endpoint_url=self.image_parser_endpoint)

        text_dict = patent_info["text"]
        image_list = patent_info["images"]
        for key, text in text_dict.items():
            text_dict[key] = self.replace_pdf_image(text, image_list)
        with open(f'{cache_path}/claim_image_text.txt', 'w') as f:
            f.write(text_dict['full'])

        return {
            "patent_id": patent_id,
            "text": text_dict,
            "claims": image_list,
        }
    
    def patent_extract(self, pdf_path):
        if self.patent_extract_method == 'google_patent':
            return self.google_patent_extract(pdf_path)
        else:
            raise NotImplementedError(f'patent_extract_method {self.patent_extract_method} is not implemented yet.')