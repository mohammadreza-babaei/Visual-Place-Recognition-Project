import os
import sys
import argparse
import torch

from glob import glob
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy

from util import read_file_preds

sys.path.append(str(Path(__file__).parent.joinpath("image-matching-models")))

from vismatch import get_matcher, available_models
from vismatch.utils import get_default_device


def fix_path(path):
    path = Path(path)

    if path.exists():
        return str(path)

    path_str = str(path)

    old_prefix = str(Path("..") / ".." / "Visual-Place-Recognition-Project")
    new_prefix = str(Path("..") / "Visual-Place-Recognition-Project")

    if path_str.startswith(old_prefix):
        fixed_path = Path(
            path_str.replace(
                old_prefix,
                new_prefix,
                1,
            )
        )

        if fixed_path.exists():
            return str(fixed_path)

    return str(path)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--preds-dir",
        type=str,
        required=True,
        help="directory with predictions of a VPR model",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="output directory of image matching results",
    )

    parser.add_argument(
        "--matcher",
        type=str,
        default="loftr",
        choices=available_models,
        help="choose your matcher",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=get_default_device(),
        choices=["cpu", "cuda"],
    )

    parser.add_argument(
        "--im-size",
        type=int,
        default=512,
        help="resize image to im_size x im_size",
    )

    parser.add_argument(
        "--num-preds",
        type=int,
        default=20,
        help="number of retrieved predictions to match",
    )

    parser.add_argument(
        "--start-query",
        type=int,
        default=-1,
        help="query index to start from",
    )

    parser.add_argument(
        "--num-queries",
        type=int,
        default=-1,
        help="number of queries to process",
    )

    return parser.parse_args()


def main(args):
    device = args.device
    matcher_name = args.matcher
    img_size = args.im_size
    num_preds = args.num_preds

    matcher = get_matcher(
        matcher_name,
        device=device,
    )

    preds_folder = args.preds_dir
    start_query = args.start_query
    num_queries = args.num_queries

    if args.out_dir is None:
        output_folder = Path(f"{preds_folder}_{matcher_name}")
    else:
        output_folder = Path(args.out_dir)

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    txt_files = glob(
        os.path.join(
            preds_folder,
            "*.txt",
        )
    )

    txt_files.sort(
        key=lambda x: int(Path(x).stem)
    )

    start_query = (
        start_query
        if start_query >= 0
        else 0
    )

    num_queries = (
        num_queries
        if num_queries >= 0
        else len(txt_files)
    )

    selected_files = txt_files[
        start_query:start_query + num_queries
    ]

    for txt_file in tqdm(selected_files):
        q_num = Path(txt_file).stem

        out_file = output_folder / f"{q_num}.torch"

        if out_file.exists():
            continue

        results = []

        q_path, pred_paths = read_file_preds(txt_file)

        q_path = fix_path(q_path)

        img0 = matcher.load_image(
            q_path,
            resize=img_size,
        )

        for pred_path in pred_paths[:num_preds]:
            pred_path = fix_path(pred_path)

            img1 = matcher.load_image(
                pred_path,
                resize=img_size,
            )

            result = matcher(
                deepcopy(img0),
                img1,
            )

            if "all_desc0" in result:
                result["all_desc0"] = None

            if "all_desc1" in result:
                result["all_desc1"] = None

            results.append(result)

        torch.save(
            results,
            out_file,
        )


if __name__ == "__main__":
    args = parse_arguments()
    main(args)