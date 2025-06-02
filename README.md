# MammAlps - A multi-view video behavior monitoring dataset of wild mammals in the Swiss Alps

<p align="center">
    <img src="resources/overview.png" alt="Overview">
</p>

MammAlps is a multimodal and multi-view dataset of wildlife behavior monitoring. Nine camera-traps were placed in the Swiss National Park, from which we curated over 14 hours of video with audio, 2D segmentation maps and 8.5 hours of individual tracks densely labeled for species and behavior.  

Along with the data, we propose two different benchmarks:
- **Benchmark I - Multimodal species and behavior recognition**: Based on 6`135 single animal clips, we propose a hierarchical and multimodal animal behavior recognition benchmark using audio, video and reference scene segmentation maps as inputs.
- **Benchmark II - Multi-view Long-term event understanding**: a second ecology-oriented benchmark aiming at identifying activities, species, number of individuals and meteorological conditions from 397 multi-view and long-term ecological events, including false positive triggers.

## Download the data
Data is available for download on [zenodo](https://doi.org/10.5281/zenodo.15040901).
```bash
wget https://zenodo.org/records/15040901/files/mammalps_v1.zip
wget https://zenodo.org/records/15040901/files/mammalps_dense_annotations.zip
```
* ``mammalps_v1.zip``: 87.2 GiB
* ``mammalps_dense_annotations.zip``: 442.8 MiB

## Installation

```bash
conda create -n mammalps python>=3.12
conda activate mammalps
pip install -r requirements.txt
```

## Usage

### Evaluation
To evaluate the performance of your model on Benchmark I, use the `eval_b1.py` script. Below is an example of how to run the script:

```bash
python eval_b1.py \
    --results_file_txt <path_to_results_file> \
    --label_names_json labels_mapping_b1.json \
    --tasks Spe ActY ActN \
    --aggregate MEAN
```

and `eval_b2.py` for Benchmark II:

```bash
python eval_b2.py \
    --results_file_txt <path_to_results_file> \
    --label_names_json labels_mapping_b2.json \
    --tasks Spe ActY Met Ind
```

**Arguments**:
* --results_file_txt: Path to the results file in .txt format. Each line should correspond to a sample with predictions and ground truth labels.
* --label_names_json: Path to the JSON file containing the mapping between label names and class IDs.
* --tasks: List of tasks to evaluate.
    * B1: [Spe, ActY, ActN]
    * B2: [Spe, ActY, Met, Ind]
* --aggregate\*: Aggregation method for test segments corresponding to a unique sample ID. Options are:
    * MEAN: Use the mean of predictions.
    * MAX: Use the maximum of predictions.

*\*aggregate option only applies to Benchmark I*

**Results file**:  
The results file (``results_file_txt``) should be a tab-separated ``.txt`` file where each line corresponds to a sample or event. Below is the format for each line:
```
<id>    <sample_id>    <test_segment_id>    <predictions_task_1> ... <predictions_task_N>   <ground_truth_task_1> ... <ground_truth_task_N>
```

* \<id>: row id.
* <sample_id>: Identifier for the sample (e.g., clip or event), corresponding to the row ids in ``test.csv``.
* <test_segment_id>: Identifier for the test segment within the sample. (always 0 for B2)
* <predictions_task_i>: Lists of predicted values for a given task (e.g., Spe, ActY, etc.).
* <ground_truth_task_i>: Lists of ground-truth values for the corresponding task

Ensure your model outputs follow this format to be compatible with the evaluation scripts.


### Model Training
We based our training pipeline off the [InterVideo](https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo1/Pretrain/VideoMAE) github repository.

### Video Processing with ToME for Benchmark II
Coming soon

### Dense annotations visualization
A simple [demo notebook](./demo/LoadDenseAnnotations.ipynb) is provided to visualize the annotated videos with bounding boxes, labels, and metadata.  
See [DATASET.md](DATASET.md) for documentation regarding the dense annotations format.  
For batch processing, you can use the `generate_visualization_videos.py` script:

```bash
python generate_visualization_videos.py \
    <input_video_folder> \
    <input_json_folder> \
    <output_video_folder> \
    --color_by action \
    --skip_existing
```

**Arguments**:
* ``<input_video_folder>``: Path to the folder containing input videos.
* ``<input_json_folder>``: Path to the folder containing JSON dense annotations.
* ``<output_video_folder>``: Path to the folder where annotated videos will be saved.
* --color_by: Specifies the annotation type for coloring bounding boxes.
* --skip_existing: Skips processing if the output video already exists.

## Citation
```bibtex
@article{gabeff2025mammalps,
      title={MammAlps: A multi-view video behavior monitoring dataset of wild mammals in the Swiss Alps},
      author={Valentin Gabeff and Haozhe Qi and Brendan Flaherty and Gencer Sumbül and Alexander Mathis and Devis Tuia},
      year={2025},
      journal={arXiv},
      doi={10.48550/arXiv.2503.18223},
}
```

## Acknowledgements
We thank members of the Mathis Group for Computational Neuroscience \& AI (EPFL) and of the Environmental Computational Science and Earth Observation Laboratory (EPFL) for their feedback and fieldwork efforts. We also thank members of the Swiss National Park monitoring team for their support and feedback. The project was approved by the Research Commission of the National Park.

## License
This code is released under the [MIT license](https://choosealicense.com/licenses/mit/)
