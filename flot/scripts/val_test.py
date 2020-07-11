import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from flot.models.scene_flow import FLOT
from torch.utils.data import DataLoader
from flot.datasets.generic import Batch


def compute_epe(est_flow, batch):
    """
    Compute EPE, accuracy and number of outliers.

    Parameters
    ----------
    est_flow : torch.Tensor
        Estimated flow.
    batch : flot.datasets.generic.Batch
        Contains ground truth flow and mask.

    Returns
    -------
    EPE3D : float
        End point error.
    acc3d_strict : float
        Strict accuracy.
    acc3d_relax : float
        Relax accuracy.
    outlier : float
        Percentage of outliers.

    """

    # Extract occlusion mask
    mask = batch["ground_truth"][0].cpu().numpy()[..., 0]

    # Flow
    sf_gt = batch["ground_truth"][1].cpu().numpy()[mask > 0]
    sf_pred = est_flow.cpu().numpy()[mask > 0]

    #
    l2_norm = np.linalg.norm(sf_gt - sf_pred, axis=-1)
    EPE3D = l2_norm.mean()

    #
    sf_norm = np.linalg.norm(sf_gt, axis=-1)
    relative_err = l2_norm / (sf_norm + 1e-4)
    acc3d_strict = (
        (np.logical_or(l2_norm < 0.05, relative_err < 0.05)).astype(np.float).mean()
    )
    acc3d_relax = (
        (np.logical_or(l2_norm < 0.1, relative_err < 0.1)).astype(np.float).mean()
    )
    outlier = (np.logical_or(l2_norm > 0.3, relative_err > 0.1)).astype(np.float).mean()

    return EPE3D, acc3d_strict, acc3d_relax, outlier


def eval_model(scene_flow, testloader):
    """
    Compute performance metrics on test / validation set.

    Parameters
    ----------
    scene_flow : flot.models.FLOT
        FLOT model to evaluate.
    testloader : flot.datasets.generic.SceneFlowDataset
        Dataset  loader.
    no_refine : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    mean_epe : float
        Average EPE on dataset.
    mean_outlier : float
        Average percentage of outliers.
    mean_acc3d_relax : float
        Average relaxed accuracy.
    mean_acc3d_strict : TYPE
        Average strict accuracy.

    """

    # Init.
    running_epe = 0
    running_outlier = 0
    running_acc3d_relax = 0
    running_acc3d_strict = 0

    #
    scene_flow = scene_flow.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for it, batch in enumerate(tqdm(testloader)):

        # Send data to GPU
        batch = batch.to(device, non_blocking=True)

        # Estimate flow
        with torch.no_grad():
            est_flow = scene_flow(batch["sequence"])

        # Perf. metrics
        EPE3D, acc3d_strict, acc3d_relax, outlier = compute_epe(est_flow, batch)
        running_epe += EPE3D
        running_outlier += outlier
        running_acc3d_relax += acc3d_relax
        running_acc3d_strict += acc3d_strict

    #
    mean_epe = running_epe / (it + 1)
    mean_outlier = running_outlier / (it + 1)
    mean_acc3d_relax = running_acc3d_relax / (it + 1)
    mean_acc3d_strict = running_acc3d_strict / (it + 1)

    print(
        "EPE;{0:e};Outlier;{1:e};ACC3DR;{2:e};ACC3DS;{3:e};Size;{4:d}".format(
            mean_epe,
            mean_outlier,
            mean_acc3d_relax,
            mean_acc3d_strict,
            len(testloader),
        )
    )

    return mean_epe, mean_outlier, mean_acc3d_relax, mean_acc3d_strict


def my_main(dataset_name, max_points, path2ckpt, test=False):
    """
    Entry point of the script.

    Parameters
    ----------
    dataset_name : str
        Dataset on which to evaluate. Either HPLFlowNet_kitti or HPLFlowNet_FT3D
        or flownet3d_kitti or flownet3d_FT3D.
    max_points : int
        Number of points in point clouds.
    path2ckpt : str
        Path to saved model.
    test : bool, optional
        Whether to use test set of validation. Has only an effect for FT3D.
        The default is False.

    Raises
    ------
    ValueError
        Unknown dataset.

    """

    # Path to current file
    pathroot = os.path.dirname(__file__)

    # Select dataset
    if dataset_name.split("_")[0].lower() == "HPLFlowNet".lower():

        # HPLFlowNet version of the datasets
        path2data = os.path.join(pathroot, "..", "data", "HPLFlowNet")

        # KITTI
        if dataset_name.split("_")[1].lower() == "kitti".lower():
            mode = "test"
            path2data = os.path.join(path2data, "KITTI_processed_occ_final")
            from flot.datasets.kitti_hplflownet import Kitti

            dataset = Kitti(root_dir=path2data, nb_points=max_points)

        # FlyingThing3D
        elif dataset_name.split("_")[1].lower() == "ft3d".lower():
            path2data = os.path.join(path2data, "FlyingThings3D_subset_processed_35m")
            from flot.datasets.flyingthings3d_hplflownet import FT3D

            mode = "test" if test else "val"
            assert mode == "val" or mode == "test", "Problem with mode " + mode
            dataset = FT3D(root_dir=path2data, nb_points=max_points, mode=mode)

        else:
            raise ValueError("Unknown dataset " + dataset_name)

    elif dataset_name.split("_")[0].lower() == "flownet3d".lower():

        # FlowNet3D version of the datasets
        path2data = os.path.join(pathroot, "..", "data", "flownet3d")

        # KITTI
        if dataset_name.split("_")[1].lower() == "kitti".lower():
            mode = "test"
            path2data = os.path.join(path2data, "kitti_rm_ground")
            from flot.datasets.kitti_flownet3d import Kitti

            dataset = Kitti(root_dir=path2data, nb_points=max_points)

        # FlyingThing3D
        elif dataset_name.split("_")[1].lower() == "ft3d".lower():
            path2data = os.path.join(path2data, "data_processed_maxcut_35_20k_2k_8192")
            from flot.datasets.flyingthings3d_flownet3d import FT3D

            mode = "test" if test else "val"
            assert mode == "val" or mode == "test", "Problem with mode " + mode
            dataset = FT3D(root_dir=path2data, nb_points=max_points, mode=mode)

        else:
            raise ValueError("Unknown dataset" + dataset_name)

    else:
        raise ValueError("Unknown dataset " + dataset_name)
    print("\n\nDataset: " + path2data + " " + mode)

    # Dataloader
    testloader = DataLoader(
        dataset,
        batch_size=1,
        pin_memory=True,
        shuffle=True,
        num_workers=6,
        collate_fn=Batch,
        drop_last=False,
    )

    # Load FLOT model
    scene_flow = FLOT(nb_iter=None)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scene_flow = scene_flow.to(device, non_blocking=True)
    file = torch.load(path2ckpt)
    scene_flow.nb_iter = file["nb_iter"]
    scene_flow.load_state_dict(file["model"])
    scene_flow = scene_flow.eval()

    # Evaluation
    epsilon = 0.03 + torch.exp(scene_flow.epsilon).item()
    gamma = torch.exp(scene_flow.gamma).item()
    power = gamma / (gamma + epsilon)
    print("Epsilon;{0:e};Power;{1:e}".format(epsilon, power))
    eval_model(scene_flow, testloader)


if __name__ == "__main__":

    # Args
    parser = argparse.ArgumentParser(description="Test FLOT.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="flownet3d_kitti",
        help="Dataset. Either HPLFlowNet_kitti or "
        + "HPLFlowNet_FT3D or flownet3d_kitti or flownet3d_FT3D.",
    )
    parser.add_argument(
        "--test", action="store_true", default=False, help="Test or validation datasets"
    )
    parser.add_argument(
        "--nb_points",
        type=int,
        default=2048,
        help="Maximum number of points in point cloud.",
    )
    parser.add_argument(
        "--path2ckpt",
        type=str,
        default="../pretrained_models/model_2048.tar",
        help="Path to saved checkpoint.",
    )
    args = parser.parse_args()

    # Launch training
    my_main(args.dataset, args.nb_points, args.path2ckpt, args.test)
