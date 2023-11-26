
import argparse,torch
from sd import reproduce_stable_diffusion_results
from clip import *

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="An example script to parse command-line arguments.")

    # Add arguments
    parser.add_argument("-o","dataset",type=str,choices=["custom","countbench"],help="choose from custom dataset or countbench")
    parser.add_argument("-t","task",type=str,choices=["classification","image_retrievel","image_gen"],help="choose the task")
    parser.add_argument("-m","model",type=str,choices=["clip_base_32","clip_base_16","clip_large_14","stable_diffusion"],help="choose the task")
    parser.add_argument("-r","ref_obj",type=str,help="name of the object being used as an reference")   

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    local_directory = "reproduced_results"

    # run image generation with stable diffucsion
    if args.task == "image_gen" and args.model == "stable_diffusion":
        pretrained_model_name="CompVis/stable-diffusion-v1-4"
        reproduce_stable_diffusion_results(local_directory,pretrained_model_name,device)
    elif args.model in ["clip_base_32","clip_base_16","clip_large_14"]:
        if args.dataset=="custom":
            # TODO: load dataset
            augmented_data = None
            if args.task == "image_retrievel" :
                image_retrievel(args.model,args.ref_obj,sample_size,augmented_data,local_directory,device=device)
            elif args.task == "classification":
                img_clf(args.model,args.ref_obj,sample_size,augmented_data,local_directory,device=device)
        elif (args.dataset=="countbench") and (args.task == "classification"):
            # TODO: load dataset
            # TODO: run on countbench
            countbench_dat = None


