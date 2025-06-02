"""
Evaluation code for Benchmark 2 - Multi-View Long-Term Event Understanding
"""

import argparse
import json
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, classification_report


def parse_line(line: str, column_names: List) -> Dict:
    elems = line.split("\t")

    output_dict = dict.fromkeys(column_names)
    for key_id, (elem, key) in enumerate(zip(elems, output_dict.keys())):
        if key_id >= 3:
            elem = np.fromstring(elem[1:-1], dtype=float, sep=",")
        else:
            elem = int(elem)
        output_dict[key] = elem

    return output_dict


def eval_b2(results_file: str, label_names: str, tasks: List) -> None:

    with open(results_file, "r") as f:
        lines = f.read().splitlines()

    print("*************************************************")
    print(f"B2 Evaluation performance for file: {results_file}")
    print("*************************************************")

    col_names = ["id", "sample_id", "test_segment_id"]
    tasks_col_preds = []
    tasks_col_labels = []
    if "ActY" in tasks:
        tasks_col_preds += ["ActY_preds"]
        tasks_col_labels += ["ActY_label"]
    if "Spe" in tasks:
        tasks_col_preds += ["Spe_preds"]
        tasks_col_labels += ["Spe_label"]
    if "Met" in tasks:
        tasks_col_preds += ["Met_preds"]
        tasks_col_labels += ["Met_label"]
    if "Ind" in tasks:
        tasks_col_preds += ["Ind_preds"]
        tasks_col_labels += ["Ind_label"]

    col_names += tasks_col_preds
    col_names += tasks_col_labels

    all_dicts = []
    for line in lines:
        all_dicts.append(parse_line(line, col_names))

    results_df = pd.DataFrame(all_dicts)
    results_df = results_df.set_index("id", drop=True)

    assert (
        len(results_df) == 86
    ), f"After aggregation, predictions for {len(results_df)} were found. 86 are expected, for each test event"

    with open(label_names, "r") as f:
        label_mapping = json.load(f)

    activity_labels = {k: v for v, k in label_mapping["activities"].items()}
    species_labels = {k: v for v, k in label_mapping["species"].items()}
    met_conds_labels = {k: v for v, k in label_mapping["env_conds"].items()}
    individuals_labels = {k: v for v, k in label_mapping["individuals"].items()}

    ## Spe: Multilabel
    if "Spe" in tasks:

        print("::Performance on Species Recognition task:: \n")

        y_true = np.vstack(results_df["Spe_label"].values).astype(int)
        preds = np.vstack(results_df["Spe_preds"])
        y_hat = np.where(preds > 0, 1, 0).astype(
            int
        )  # i.e. threshold of 0.5 after sigmoid operation

        print(
            "Spe Classification report: \n",
            classification_report(
                y_true,
                y_hat,
                labels=list(species_labels.keys()),
                target_names=species_labels.values(),
                digits=3,
                zero_division=0,
            ),
        )
        print(
            "Spe mAP per class: ",
            *zip(
                species_labels.values(),
                np.round(average_precision_score(y_true, preds, average=None), 3),
            ),
            sep="\n\t",
        )
        print(
            "Spe mAP (macro avg): \n\t",
            np.round(average_precision_score(y_true, preds, average="macro"), 3),
        )
        print("*************************************************")

    ## ActY: Multilabel
    if "ActY" in tasks:

        print("::Performance on Activity Recognition task:: \n")

        y_true = np.vstack(results_df["ActY_label"].values).astype(int)
        preds = np.vstack(results_df["ActY_preds"])
        y_hat = np.where(preds > 0, 1, 0).astype(
            int
        )  # i.e. threshold of 0.5 after sigmoid operation

        print(
            "ActY Classification report: \n",
            classification_report(
                y_true,
                y_hat,
                labels=list(activity_labels.keys()),
                target_names=activity_labels.values(),
                digits=3,
                zero_division=0,
            ),
        )
        print(
            "ActY mAP per class: ",
            *zip(
                activity_labels.values(),
                np.round(average_precision_score(y_true, preds, average=None), 3),
            ),
            sep="\n\t",
        )
        print(
            "ActY mAP (macro avg): \n\t",
            np.round(average_precision_score(y_true, preds, average="macro"), 3),
        )
        print("*************************************************")

    ## Met: Multiclass
    if "Met" in tasks:

        print("::Performance on Meteorological Condition Recognition task:: \n")

        y_true = np.argmax(np.vstack(results_df["Met_label"].values), axis=1)
        preds = np.vstack(results_df["Met_preds"])
        y_hat = np.argmax(preds, axis=1)

        print(
            "Met Classification report: \n",
            classification_report(
                y_true,
                y_hat,
                labels=list(met_conds_labels.keys()),
                target_names=met_conds_labels.values(),
                digits=3,
                zero_division=0,
            ),
        )
        print(
            "Met mAP per class: ",
            *zip(
                met_conds_labels.values(),
                np.round(average_precision_score(y_true, preds, average=None), 3),
            ),
            sep="\n\t",
        )
        print(
            "Met mAP (macro avg): \n\t",
            np.round(average_precision_score(y_true, preds, average="macro"), 3),
        )
        print("*************************************************")

    ## Ind: Multiclass
    if "Ind" in tasks:

        print("::Performance on Number of Individuals Recognition task:: \n")

        y_true = np.argmax(np.vstack(results_df["Ind_label"].values), axis=1)
        preds = np.vstack(results_df["Ind_preds"])
        y_hat = np.argmax(preds, axis=1)

        print(
            "Ind Classification report: \n",
            classification_report(
                y_true,
                y_hat,
                labels=list(individuals_labels.keys()),
                target_names=individuals_labels.values(),
                digits=3,
                zero_division=0,
            ),
        )
        print(
            "Ind mAP per class: ",
            *zip(
                individuals_labels.values(),
                np.round(average_precision_score(y_true, preds, average=None), 3),
            ),
            sep="\n\t",
        )
        print(
            "Ind mAP (macro avg): \n\t",
            np.round(average_precision_score(y_true, preds, average="macro"), 3),
        )
        print(
            "Ind mAP (macro) 2+",
            np.round(average_precision_score(y_true, preds, average=None), 3)[
                [2, 3]
            ].mean(),
        )
        print("*************************************************")

    ## Avg.
    print("Per-class average performance on all recognition tasks: \n")

    results_df["labels"] = results_df.apply(
        lambda d: np.concatenate([d[c] for c in tasks_col_labels]), axis=1
    )
    results_df["preds"] = results_df.apply(
        lambda d: np.concatenate([d[c] for c in tasks_col_preds]), axis=1
    )

    y_true = np.vstack(results_df["labels"].values).astype(int)
    preds = np.vstack(results_df["preds"].values)

    print(
        "Avg. mAP (macro avg): \n\t",
        np.round(average_precision_score(y_true, preds, average="macro"), 3),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluation code for Benchmark 1: Multimodal species and behavior recognition"
    )
    parser.add_argument(
        "--results_file_txt", type=str, required=True, help="results file to evaluate"
    )
    parser.add_argument(
        "--label_names_json",
        type=str,
        required=True,
        help="Json dictionnary containing the mapping between label names and class ids",
    )
    parser.add_argument(
        "--tasks", type=str, choices=["Spe", "ActY", "Met", "Ind"], nargs="+"
    )

    args = parser.parse_args()

    eval_b2(args.results_file_txt, args.label_names_json, args.tasks)
