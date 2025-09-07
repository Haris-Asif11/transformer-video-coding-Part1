from pathlib import Path
import time
import sys
import torch
import argparse

from detectron2.utils.logger import setup_logger
from detectron2.evaluation import CityscapesInstanceEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader, get_detection_dataset_dicts, print_instances_class_histogram, DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.cityscapes import load_cityscapes_instances





DATASET_NAME = "cityscapes_sequence"

cfg_file = "configs/Cityscapes/mask_rcnn_R_50_FPN.yaml"
model_path = "model_final_af9cf5.pkl"

def load_cityscapes_model(configFile=cfg_file, modelPath=model_path, colorFormat='BGR', returnModel=True, batchSize=None, numWorkers=8, device="cpu"):
    import sys
    #sys.path.insert(0, '/home/windsheimer/detectron2/')
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    cfg = get_cfg()
    cfg.merge_from_file(configFile)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # set threshold for this model
    cfg.MODEL.WEIGHTS = modelPath
    cfg.SOLVER.BASE_LR = 0.02
    if batchSize:
        cfg.SOLVER.BATCH_SIZE = batchSize
        cfg.SOLVER.IMS_PER_BATCH = batchSize
    cfg.DATALOADER.NUM_WORKERS = numWorkers
    cfg.INPUT.FORMAT = colorFormat
    cfg.MODEL.DEVICE = device
    if returnModel:
        predictor = DefaultPredictor(cfg)
        return predictor, cfg
    else:
        return None, cfg
#/SHARED_FILES/transfer/stud/windsheimer/4Haris/gt_gen/val
#/home/ac35anos/PycharmProjects/detectron2_new/datasets/cityscapes/leftImg8bit/val
#/home/ac35anos/PycharmProjects/detectron2_new/output_test25_100k/val
#/SHARED_FILES/sequences/Autonomous_driving/Cityscapes/gtFine/val
def parse_args(argv):
    parser = argparse.ArgumentParser(description="Script for evaluation of (un-)compressed Cityscapes sequences for instance and semantic segmentation")
    parser.add_argument("-s", "--source", type=str, default="/home/ac35anos/PycharmProjects/detectron2_new/output_test31_10k/val", help="Path to sequences")
    parser.add_argument("-g", "--gt-path", type=str, default="/SHARED_FILES/sequences/Autonomous_driving/Cityscapes/gtFine/val", help="Path to (pseudo) ground truth.")
    parser.add_argument("-m", "--mode", type=str, default="instance", choices=["instance", "semantic"], help="Modus: either instance or semantic segmentation.")
    parser.add_argument("-d", "--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Usage of GPU.")
    return parser.parse_args(argv)

def main(argv):
    args = parse_args(argv)
    if args.mode == 'instance':
        
        def load_cityscapes_instances_wrapper():
            return load_cityscapes_instances(args.source, args.gt_path, from_json=False)
        
        logger = setup_logger()
        print("Eval Cityscapes start")
        start0 = time.time()
        DatasetCatalog.register(DATASET_NAME, load_cityscapes_instances_wrapper)
        end = time.time()
        print("Time: DatasetCatalog register ", end - start0)
        start = time.time()
        MetadataCatalog.get(DATASET_NAME).thing_classes = MetadataCatalog.get("cityscapes_fine_instance_seg_test").thing_classes
        MetadataCatalog.get(DATASET_NAME).gt_dir = args.gt_path
        print_instances_class_histogram(get_detection_dataset_dicts(DATASET_NAME), MetadataCatalog.get(DATASET_NAME).thing_classes)
        end = time.time()
        print("Time: Set config ", end - start)
        start = time.time()
        predictor, cfg = load_cityscapes_model(device=args.device)
        evaluator = CityscapesInstanceEvaluator(DATASET_NAME)
        dataloader = build_detection_test_loader(cfg, DATASET_NAME)
        end = time.time()
        print("Time: Load predictor and dataloader ", end - start)
        start = time.time()
        output = inference_on_dataset(predictor.model, dataloader, evaluator)
        print(output)
        end = time.time()
        print("Time: Inference ", end - start)
        print("Time: Total ", end - start0)
        print("Finished!")
        return output
    elif args.mode == "semantic":
        print('Only accepting Instance')



if __name__ == "__main__":
    main(sys.argv[1:])