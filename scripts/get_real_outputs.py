import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.summarize import summarize

def get_real_outputs():
    ckpt_path = "models/checkpoint_epoch_10.pt"
    
    samples = [
        "def sina_xml_to_url_list(xml_data):\n    rawurl = []\n    dom = parseString(xml_data)\n    for node in dom.getElementsByTagName('durl'):\n        url = node.getElementsByTagName('url')[0]\n        rawurl.append(url.childNodes[0].data)\n    return rawurl",
        "def sprint(text, *colors):\n    return \"\\33[{}m{content}\\33[{}m\".format(\";\".join([str(color) for color in colors]), RESET, content=text) if IS_ANSI_TERMINAL and colors else text",
        "def ckplayer_get_info_by_xml(ckinfo):\n    e = ET.XML(ckinfo)\n    video_dict = {'title': '', 'links': [], 'size': 0, 'flashvars': ''}\n    dictified = dictify(e)['ckplayer']\n    return video_dict"
    ]

    print("\n=== START REAL INFERENCE ===")
    for i, s in enumerate(samples):
        # The summarize function handles tokenization, model loading, and greedy decoding
        out = summarize(s, ckpt_path)
        print(f"SAMPLE_{i}_OUT: {out}")
    print("=== END REAL INFERENCE ===\n")

if __name__ == "__main__":
    get_real_outputs()
