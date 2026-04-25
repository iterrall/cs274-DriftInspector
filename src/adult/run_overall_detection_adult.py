from argparse import ArgumentParser


from config import *
import os
from glob import glob
import sys

sys.path.append("..")


def compute_store_overall_stats(
    tgt,
    config_experiment,
    base_output_dir,
    save_results=True,
    overall_detectors_args={},
):

    from adult.overall_drift_global import (
        read_experiment,
        init_detectors,
        get_cm_detections,
    )

    tgt_values = read_experiment(tgt)
    if tgt_values is None:
        # The object is not available
        return

    altered = tgt_values["altered"]
    y_trues = tgt_values["y_trues"]
    y_preds = tgt_values["y_preds"]
    sg = tgt_values["sg"]
    subgroup_config_name = tgt_values["subgroup_config_name"]

    # Total altered
    altered_sg_batch = [
        (altered[i] == True).astype(int).sum() for i in range(len(altered))
    ]
    altered_sg = sum(altered_sg_batch)

    # Initialize the detectors
    detectors_dict, overall_detectors_args = init_detectors(overall_detectors_args)

    # Avoid recomputing if all the results are already available
    exist_all = True
    for detector_name in detectors_dict:
        basename_detector = detector_name.split("_")[0]
        config_detector = (
            detector_name.split("_")[1] if len(detector_name.split("_")) > 1 else "def"
        )
        output_filename = os.path.join(
            base_output_dir,
            config_experiment,
            basename_detector,
            config_detector,
            f"overalldrift-{subgroup_config_name}.pkl",
        )
        if os.path.exists(output_filename) == False:
            # At least one does not exist
            exist_all = False
            break
    if exist_all == True:
        # If all are are available, return. We do not recomputed it again
        return

    # Initialize dictionaries to store the results
    detector_warnings = {detector_name: {} for detector_name in detectors_dict}
    detector_detected = {detector_name: {} for detector_name in detectors_dict}
    overall_drift_result = {detector_name: {} for detector_name in detectors_dict}

    for batch_idx in range(len(y_trues)):
        # hddm_a, eddm: Whether the last sample analyzed was correctly classified or not. 1 indicates an error (miss-classification).
        errors_b = (y_trues[batch_idx] != y_preds[batch_idx]).astype(int)

        # ADWIN: 0: Means the learners prediction was wrong, 1: Means the learners prediction was correct
        corrects_b = (y_trues[batch_idx] == y_preds[batch_idx]).astype(int)

        # We dot this oly for  "chi2" or "fet"  detectors:
        if batch_idx == 0:
            # We initialize chi and fet for the entire batch
            # overall_detectors_args['chi2'] and overall_detectors_args['fet']  is the p-value threshold for the chi-square test
            if "chi2" in detectors_dict:
                from alibi_detect.cd import ChiSquareDrift

                chi = ChiSquareDrift(errors_b, overall_detectors_args["chi2"])
                detectors_dict["chi2"] = chi
            if "fet" in detectors_dict:
                from alibi_detect.cd import FETDrift

                fet = FETDrift(errors_b, overall_detectors_args["fet"])
                detectors_dict["fet"] = fet
        else:
            for detector_name, detector in detectors_dict.items():
                if detector_name == "chi2" or detector_name == "fet":
                    # We evaluate chi or fet for the entire batch
                    preds = detectors_dict[detector_name].predict(errors_b)
                    is_drift = preds["data"]["is_drift"]
                    if is_drift:
                        # Add detected change to the dictionary
                        # We say that it detect a drift for all samples in the batch
                        detector_detected[detector_name][batch_idx] = [
                            1 for i in range(len(errors_b))
                        ]

        # For the other approaches, we iterate one sample at the time
        for i in range(len(errors_b)):
            for detector_name, detector in detectors_dict.items():
                if detector_name == "chi2" or detector_name == "fet":
                    # We skip chi2 and fet has we do the evaluation for the entire batch
                    continue
                elif detector_name[0:5] == "adwin":
                    detector.add_element(corrects_b[i])
                else:
                    detector.add_element(errors_b[i])

                # Start detecting change after the first batch
                if batch_idx >= 1:

                    if detector.detected_warning_zone():
                        # Add warning zone to the dictionary

                        if batch_idx not in detector_warnings[detector_name]:
                            detector_warnings[detector_name][batch_idx] = []
                        detector_warnings[detector_name][batch_idx].append(i)
                        # print('Warning zone has been detected in data: ' + str(errors_b[i]) + ' - of index: ' + str(i))
                    if detector.detected_change():
                        # Add detected change to the dictionary
                        # print(f"{detector_name} - Change has been detected in batch_idx: {batch_idx} - of index: {i}")
                        if batch_idx not in detector_detected[detector_name]:
                            detector_detected[detector_name][batch_idx] = []
                        detector_detected[detector_name][batch_idx].append(i)

    overall_drift_result = {}

    # Store subgroup results
    for detector_name in detector_detected:
        basename_detector = detector_name.split("_")[0]
        config_detector = (
            detector_name.split("_")[1] if len(detector_name.split("_")) > 1 else "def"
        )
        overall_drift_result = {
            "subgroup": sg,
            "altered_sg": altered_sg,
            f"{basename_detector}_warnings": get_cm_detections(
                altered_sg_batch, detector_warnings[detector_name]
            ),
            f"{basename_detector}_detected": get_cm_detections(
                altered_sg_batch, detector_detected[detector_name]
            ),
            f"{basename_detector}_detected_batch": detector_detected[detector_name],
            "file_basename": os.path.basename(tgt),
            f"{basename_detector}_num_pts_detected": sum(
                [len(v) for v in detector_detected[detector_name].values()]
            ),
        }
        if save_results:
            import pathlib
            import pickle

            output_dir = os.path.join(
                base_output_dir, config_experiment, basename_detector, config_detector
            )

            p = pathlib.Path(output_dir)
            p.mkdir(parents=True, exist_ok=True)

            output_filename = os.path.join(
                output_dir, f"overalldrift-{subgroup_config_name}.pkl"
            )

            with open(output_filename, "wb") as f:
                pickle.dump(overall_drift_result, f)


if __name__ == "__main__":

    parser = ArgumentParser(description="Compute dataset statistics")

    parser.add_argument(
        "--dataset", help="Dataset to use", default="adult", required=False, type=str
    )

    parser.add_argument(
        "--output_dir_name",
        help="Directory where the results are stored",
        default="results-drift-overall",
        required=False,
        type=str,
    )

    args = parser.parse_args()

    dataset = args.dataset
    output_dir_name = args.output_dir_name

    if dataset == "adult":
        ckpt_dir = "/data2/fgiobergia/drift-experiments/"

        checkpoint = "xgb-adult"
        noise_frac = 0.5
    elif dataset == "celeba":
        ckpt_dir = "/home/fgiobergia/div-mitigation/models-ckpt/sup-wise"
        checkpoint = "resnet50"
        noise_frac = 1.0
    else:
        raise ValueError(f"Dataset {dataset} not supported")

    base_output_dir = os.path.join(output_dir_name, f"{checkpoint}")

    print(f"Output base directory: {base_output_dir}")

    neg = glob(os.path.join(ckpt_dir, f"{checkpoint}-noise-0.00-support-*pkl"))
    pos = glob(
        os.path.join(ckpt_dir, f"{checkpoint}-noise-{noise_frac:.2f}-support-*pkl")
    )

    from tqdm import tqdm

    overall_detectors_args = {
        "kswin_window_size": [300],  # 500, 1000, 2000, 8000, 2000
    }

    print("Positive")
    for tgt_idx, tgt in enumerate(tqdm(pos)):
        compute_store_overall_stats(
            tgt,
            "pos",
            base_output_dir,
            save_results=True,
            overall_detectors_args=overall_detectors_args,
        )

    print("Negative")
    for tgt_idx, tgt in enumerate(tqdm(neg)):
        compute_store_overall_stats(
            tgt,
            "neg",
            base_output_dir,
            save_results=True,
            overall_detectors_args=overall_detectors_args,
        )