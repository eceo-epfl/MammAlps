"""
Evaluation code for Benchmark 1 - Multimodal species and behavior recognition
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


def aggregate_mean_predictions(row):
    if isinstance(row.iloc[0], np.ndarray):
        return np.mean(np.vstack(row), axis=0)
    else:
        return row.iloc[0]


def aggregate_max_predictions(row):
    if isinstance(row.iloc[0], np.ndarray):
        return np.max(np.vstack(row), axis=0)
    else:
        return row.iloc[0]


def eval_b1(results_file: str, label_names: str, tasks: List, aggregate: str) -> None:

    with open(results_file, "r") as f:
        lines = f.read().splitlines()

    print("*************************************************")
    print(f"B1 Evaluation performance for file: {results_file}")
    print("*************************************************")

    col_names = ["id", "sample_id", "test_segment_id"]
    tasks_col_preds = []
    tasks_col_labels = []
    if "ActY" in tasks:
        tasks_col_preds += ["ActY_preds"]
        tasks_col_labels += ["ActY_label"]
    if "ActN" in tasks:
        tasks_col_preds += ["ActN_preds"]
        tasks_col_labels += ["ActN_label"]
    if "Spe" in tasks:
        tasks_col_preds += ["Spe_preds"]
        tasks_col_labels += ["Spe_label"]

    col_names += tasks_col_preds
    col_names += tasks_col_labels

    all_dicts = []
    for line in lines:
        all_dicts.append(parse_line(line, col_names))

    results_df = pd.DataFrame(all_dicts)
    results_df = results_df.set_index("id", drop=True)

    if aggregate == "MEAN":
        aggreg_op = aggregate_mean_predictions
    elif aggregate == "MAX":
        aggreg_op = aggregate_max_predictions
    else:
        raise NotImplementedError(
            f"Choice of aggregation {aggregate} is not implemented"
        )

    results_df = (
        results_df.groupby("sample_id").agg(aggreg_op).drop("test_segment_id", axis=1)
    )

    assert (
        len(results_df) == 1244
    ), f"After aggregation, predictions for {len(results_df)} were found. 1244 are expected, for each test clip"

    with open(label_names, "r") as f:
        label_mapping = json.load(f)

    activity_labels = {k: v for v, k in label_mapping["activities"].items()}
    action_labels = {k: v for v, k in label_mapping["actions"].items()}
    species_labels = {k: v for v, k in label_mapping["species"].items()}

    ## Spe: Multiclass
    if "Spe" in tasks:

        print("::Performance on Species Recognition task:: \n")

        y_true = np.argmax(np.vstack(results_df["Spe_label"].values), axis=1)
        preds = np.vstack(results_df["Spe_preds"])
        y_hat = np.argmax(preds, axis=1)

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

    ## ActY: Multiclass
    if "ActY" in tasks:

        print("::Performance on Activity Recognition task:: \n")

        y_true = np.argmax(np.vstack(results_df["ActY_label"].values), axis=1)
        preds = np.vstack(results_df["ActY_preds"])
        y_hat = np.argmax(preds, axis=1)

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

    ## ActN: Multilabel
    if "ActN" in tasks:

        print("::Performance on Action Recognition task:: \n")

        y_true = np.vstack(results_df["ActN_label"].values).astype(int)
        preds = np.vstack(results_df["ActN_preds"])
        y_hat = np.where(preds > 0, 1, 0).astype(
            int
        )  # i.e. threshold of 0.5 after sigmoid operation

        print(
            "ActN Classification report: \n",
            classification_report(
                y_true,
                y_hat,
                labels=list(action_labels.keys()),
                target_names=action_labels.values(),
                digits=3,
                zero_division=0,
            ),
        )
        print(
            "ActN mAP per class: ",
            *zip(
                action_labels.values(),
                np.round(average_precision_score(y_true, preds, average=None), 3),
            ),
            sep="\n\t",
        )
        print(
            "ActN mAP (macro avg): \n\t",
            np.round(average_precision_score(y_true, preds, average="macro"), 3),
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
    parser.add_argument("--tasks", type=str, choices=["Spe", "ActY", "ActN"], nargs="+")
    parser.add_argument(
        "--aggregate",
        type=str,
        choices=["MEAN", "MAX"],
        required=True,
        help="Operation to aggregate the different test_segment_id corresponding to a unique sample_id",
    )

    args = parser.parse_args()

    eval_b1(args.results_file_txt, args.label_names_json, args.tasks, args.aggregate)
