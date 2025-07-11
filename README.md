
# Environment Setup

1. Create and activate Conda virtual environment
```bash
conda create -n 3dgsg python=3.8
conda activate 3dgsg
```

2. Install PyTorch with CUDA 11.8 support
```bash
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```

3. Install submodule dependencies
```bash
cd submodules

#  Install diff-gaussian-rasterization
cd diff-gaussian-rasterization
pip install -e .
cd ..

# Install simple-knn
cd simple-knn
pip install -e .
cd ..
cd ..
```

4. Install Python dependencies
```bash
pip install chamferdist==1.0.3
pip install plyfile==1.0.3
pip install tqdm
pip install opencv-python==4.10.0.84
pip install openai-clip==1.0.1
pip install pandas==2.0.3
pip install matplotlib
pip install cupy-cuda11x==12.3.0
pip install open3d==0.18.0
```
5. Install Segment Anything module
```bash
cd segment-anything-main/
pip install -e .
cd ..
```

# Download Related Data
## Download Demo Data
We provide a demo, you can download the demo dataset through the following link: [📎 Download Link (Google Drive)](https://drive.google.com/file/d/1QA4R5WbqJ-ko5lVPK9xYuy_9l8BSPYxF/view?usp=drive_link)

The file structure is as follows:
```bash
data/
├── images/       # Store the original image, such as frame000.jpg
│   ├── frame000.jpg       
│   ├── framexxx.jpg    
│   ├── ………………………………    
├── results/      # Store text matching mask results
│   ├── THERMOS_clip_results.csv   
├── sparse/       # Stores the sparse reconstruction data output by COLMAP
│   ├── 0/
│   │   ├── cameras.bin     
│   │   ├── images.bin      
│   │   ├── points3D.biun   
│   │   ├── cameras.txt     
│   │   ├── images.txt      
│   │   ├── points3D.txt
│   │   ├── points3D.ply
```

## Download pre-trained weights (Checkpoints)
1. Create a ckpts folder in your project root
```bash
mkdir ckpts
```

2. Please download the checkpoint file from the following Google Drive link:

 [📎 Download Link (Google Drive)](https://drive.google.com/drive/folders/1ZNgnIIGx0VIrYRVuQ8WAhessbpIiqKe5)

After downloading, move the .pth file into the ckpts/ folder in your project root.

# Train

## Usage Guide  
⚠️ Note: The paths below are based on the developer’s environment. Please modify them to match your own system.

## Step 1: 3D-GS data
1. Train 3D-GS based on data
```bash
python train.py \
  -s /media/djape/EXData/Grounding/data \
  -m /media/djape/EXData/Grounding/data/output
```

2. Rendering Views
```bash
python render_views.py \
  -s /media/djape/EXData/Grounding/data \
  -m /media/djape/EXData/Grounding/data/output \
  --target_width 3819 \
  --target_height 2146 \
  --restore_checkpoint /media/djape/EXData/Grounding/data/output/chkpnt30000.pth \
  --render_output_folder /media/djape/EXData/Grounding/data/render_all
```

3. Selecting Rendered Views  
Since there are many training perspectives, 90 perspectives are sampled here.
```bash
python select_images.py \
  --input_folder /media/djape/EXData/Grounding/data/render_all \
  --output_folder /media/djape/EXData/Grounding/data/render \
  --num_selected 90
```

## Step 2: Mask data
⚠️ **Important: select_images.py and extract_object.py must be run at the same to save time ! ! !**
1. Generate Masks from Rendered Views using SAM  
⚠️ Note: Make sure mask_gen.py is run inside the segment-anything-main folder.
```bash
cd segment-anything-main
python select_images.py \
  --input_folder /media/djape/EXData/Grounding/data/render_all \
  --output_folder /media/djape/EXData/Grounding/data/render \
  --num_selected 90
cd ..
```

2. Extract Foreground Objects  
This script will extract the target image region from the mask
```bash
python extract_object.py \
  --base-dir /media/djape/EXData/Grounding/data/render \
  --target-size 3819 2146
```

## Step 3: Semantic mask extraction
⚠️ Important: Make sure the results/ folder already exists inside your /data directory before running the script.

1. Obtain semantically relevant mask information based on the text prompt.  
The semantic prompt used in the demo data is:  
"A compact, cylindrical thermos jug with a wide body and a small spout. It features a matte black top with a rounded, ergonomic handle, and a light grey middle section wrapping around the body. The top has a violet push-button for pouring and two small holes for ventilation."
```bash
python clip_match.py \
  --text "A compact, cylindrical thermos jug with a wide body and a small spout. It features a matte black top with a rounded, ergonomic handle, and a light grey middle section wrapping around the body. The top has a violet push-button for pouring and two small holes for ventilation. " \
  --root-folder /media/djape/EXData/Grounding/data/render \
  --topk 3 \
  --save-csv /media/djape/EXData/Grounding/data/results/THERMOS_clip_results.csv
```

2. Use SDM to obtain consistent masks.
```bash
cd SDM
python main.py \
  --csv_path "/media/djape/EXData/Grounding/data/results/THERMOS_clip_results.csv" \
  --text "A compact, cylindrical thermos jug with a wide body and a small spout. It features a matte black top with a rounded, ergonomic handle, and a light grey middle section wrapping around the body. The top has a violet push-button for pouring and two small holes for ventilation." \
  --output_root "/media/djape/EXData/Grounding/data" \
  --render_path "/media/djape/EXData/Grounding/data/render" \
  --topn_mask 150 \
  --topn_clip 100 \
  --topn_feature 10
cd ..
```

## Step 4: Obtaining a point cloud with consistent viewpoint
Use AVGM to obtain the initial Filtering Gaussian point cloud  
A number of commands need to be executed. Please run them sequentially.  
```bash
cd AVGM
python pick_view.py \
  --folder-path /media/djape/EXData/Grounding/data/results/views \
  --images-file /media/djape/EXData/Grounding/data/sparse/0/images.txt \
  --top-x 1

python pick_find_use.py \
  --base-dir /media/djape/EXData/Grounding/data/results/views \
  --render-dir /media/djape/EXData/Grounding/data/render

python find_cuda_use.py \
  --cameras-path /media/djape/EXData/Grounding/data/sparse/0/cameras.txt \
  --images-path /media/djape/EXData/Grounding/data/sparse/0/images.txt \
  --point-cloud-path /media/djape/EXData/Grounding/data/output/point_cloud/iteration_30000/point_cloud.ply

python intersection.py \
    --folder /media/djape/EXData/Grounding/data/results \
    --output intersection.npy \
    --batch-size 2000
cd ..
```

Finally, the views with large disparities that are semantically consistent with the text description will be extracted and saved in the **data/results/Views/FindViewX** folder.  
At the same time, the initially filtered point cloud is saved as intersection.npy in the **data/results** folder.

## Step 5: Get the result of 3D-GS Grounding
1. Use RANN to predict the threshold. Please **record the threshold** according to the output.
```bash
python RANN_predict.py \
  -s /media/djape/EXData/Grounding/data \
  -m /media/djape/EXData/Grounding/data/output \
  --pointcloud_path /media/djape/EXData/Grounding/data/results/intersection.npy
```

2. Render the results to the **data/output** folder based on the threshold, filtered point cloud, and related information.  
Please fill the threshold obtained from the previous step into the command.

```bash
python GSG_train.py -s /media/djape/EXData/Grounding/data -m /media/djape/EXData/Grounding/data/output \
  --checkpoint_path /media/djape/EXData/Grounding/data/output/chkpnt30000.pth \
  --intersection_path /media/djape/EXData/Grounding/data/results/intersection.npy \
  --threshold 0.0414
```

The results are saved in data/output/point_cloud/iteration_GSG. Please use 3D Gaussian Viewer to view it.











