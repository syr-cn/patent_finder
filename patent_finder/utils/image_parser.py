import os
import sys
import io
import json
import requests
import base64
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

def image_to_base64(image):
    image = image.convert('RGB')
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    return base64.b64encode(img_byte_arr.getvalue()).decode("ascii")

def extract_image(image_paths, url, timeout=120, batch_size=128):
    """
    output example: 
    {
        "beam_idx": [
            0
        ],
        "caption": [
            "*CSC1NN=CN=1.S(*SC1NN=CN=1)(=O)=O.*SC1=NN=C(S*)N1.C1=NN=C(S*)N1.C1=CC(O)=NN1.C1=C(S(=O)*)NN=N1<sep><a>0:<dum></a><a>9:<dum></a><a>18:<dum></a><a>25:<dum></a><a>32:<dum></a><a>43:<dum></a>"
        ],
        "drawing": [
            ""
        ],
        "markush": [
            true
        ],
        "score": [
            0.8479739320388734
        ],
        "smi": [
            "*CSc1ncn[nH]1.*S(=O)c1cnn[nH]1.*Sc1nnc(S*)[nH]1.*Sc1nnc[nH]1.O=[SH](=O)*Sc1ncn[nH]1.Oc1cc[nH]n1"
        ],
        "sru": [
            false
        ]
    }
    """
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    if len(image_paths) > batch_size:
        print(f'Extracting {len(image_paths)} images in batches of {batch_size}')
        image_data_list = []
        for idx in range(0, len(image_paths), batch_size):
            image_data_list += extract_image(image_paths[idx:idx+batch_size], url, timeout, batch_size)
        return image_data_list

    valid_paths = [image_path for image_path in image_paths if os.path.exists(image_path)]
    response = requests.post(
        url,
        json={"batch_image": [image_to_base64(Image.open(image_path)) for image_path in valid_paths]},
        timeout=timeout,
    )
    print(f'Extracted {len(image_paths)} images, response: {response.status_code}')
    final_response = response.json()

    try:
        data = final_response['data']
        keys = list(data.keys())
        value_length = len(data[keys[0]])
        data_list = []
        assert len(valid_paths) == value_length
        if not all(len(values) == value_length for values in data.values()):
            raise ValueError("All value lists must have the same length")
        for i in range(value_length):
            data_dict = {key: data[key][i] for key in keys}
            data_list.append(data_dict)

        valid_data_map = dict(zip(valid_paths, data_list))
        total_data_list = [valid_data_map.get(image_path, {}) for image_path in image_paths]
        return total_data_list
    except:
        print('Failed to convert image to smiles.')
        return None

def download_image(image_link, image_path, cache_path):
    if os.path.exists(image_path):
        return image_path
    img_response = requests.get(image_link, timeout=10)
    if img_response.status_code == 200:
        with open(image_path, 'wb') as file:
            file.write(img_response.content)
        print(f'{image_path} downloaded successfully.')
    else:
        print(f'Failed to download {image_path} from {image_link}')
        
        # Update current image link to 404 list
        with open(f'{cache_path}/url_404_list.json', 'r') as f:
            links_404_list = json.load(f)
        if not image_link in links_404_list:
            links_404_list.append(image_link)
        with open(f'{cache_path}/url_404_list.json', 'w') as f:
            json.dump(links_404_list, f, indent=4)

def download_images(image_links, cache_path, endpoint_url, max_workers=10):
    image_paths = [f'{cache_path}/image_{idx+1}.png' for idx in range(len(image_links))]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_image, link, path, cache_path) for path, link in zip(image_paths, image_links)]
        for future in as_completed(futures):
            future.result()

    exist_paths = [path for path in image_paths if os.path.exists(path)]
    # Parse local Image
    image_data_list = extract_image(exist_paths, url=endpoint_url)
    image_data_dict = {img_path:img_data for img_path, img_data in zip(exist_paths, image_data_list)}

    image_dicts = []
    for link, image_path in zip(image_links, image_paths):
        if image_path in image_data_dict:
            image_data = image_data_dict[image_path]
        else:
            image_data = {'markush': False, 'smi': '404 Image Not Found', 'sru': False, 'caption': '404 Image Not Found', 'score': 0.0}
        
        image_json_path = image_path.replace('.png', '.json')
        image_data['link'] = link
        with open(image_json_path, 'w') as f:
            json.dump(image_data, f, indent=4)
        
        image_data['path'] = image_path
        image_data['json_path'] = image_json_path
        image_dicts.append(image_data)
    sys.stdout.flush()

    link_order = {link: idx for idx, link in enumerate(image_links)}
    image_dicts = sorted(image_dicts, key=lambda x: link_order[x['link']])
    return image_dicts