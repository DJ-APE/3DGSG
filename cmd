conda create -n 3dgsg python=3.8
conda activate 3dgsg

conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia

cd submodules
cd diff-gaussian-rasterization
pip install e .
cd ..
cd simple-knn
pip install e .
cd ..
cd ..

pip install chamferdist==1.0.3
pip install plyfile==1.0.3
pip install tqdm
pip install opencv-python==4.10.0.84
pip install openai-clip==1.0.1
pip install pandas==2.0.3
pip install matplotlib
pip install cupy-cuda11x==12.3.0
pip install open3d==0.18.0




cd segment-anything-main/
pip install e .


训练
python train.py -s /media/djape/EXData/Grounding/data -m /media/djape/EXData/Grounding/data/output

渲染多视角图像
python render_views.py \
  -s /media/djape/EXData/Grounding/data \
  -m /media/djape/EXData/Grounding/data/output \
  --target_width 3819 \
  --target_height 2146 \
  --restore_checkpoint /media/djape/EXData/Grounding/data/output/chkpnt30000.pth \
  --render_output_folder /media/djape/EXData/Grounding/data/render_all

采样
python select_images.py \
  --input_folder /media/djape/EXData/Grounding/data/render_all \
  --output_folder /media/djape/EXData/Grounding/data/render \
  --num_selected 90


生成mask
确保其在sam文件夹内
python mask_gen.py \
    --input-folder /media/djape/EXData/Grounding/data/render \
    --output-folder /media/djape/EXData/Grounding/data/render \
    --checkpoint /media/djape/EXData/Grounding/3DGSG/ckpts/sam_vit_h_4b8939.pth \
    --model-type vit_h \
    --script scripts/amg.py

同步运行，提取图像
python extract_object.py \
  --base-dir /media/djape/EXData/Grounding/data/render \
  --target-size 3819 2146

创建results文件夹！！！！！！

python clip_match.py \
  --text "A compact, cylindrical thermos jug with a wide body and a small spout. It features a matte black top with a rounded, ergonomic handle, and a light grey middle section wrapping around the body. The top has a violet push-button for pouring and two small holes for ventilation. " \
  --root-folder /media/djape/EXData/Grounding/data/render \
  --topk 3 \
  --save-csv /media/djape/EXData/Grounding/data/results/THERMOS_clip_results.csv

python main.py \
  --csv_path "/media/djape/EXData/Grounding/data/results/THERMOS_clip_results.csv" \
  --text "A compact, cylindrical thermos jug with a wide body and a small spout. It features a matte black top with a rounded, ergonomic handle, and a light grey middle section wrapping around the body. The top has a violet push-button for pouring and two small holes for ventilation." \
  --output_root "/media/djape/EXData/Grounding/data" \
  --render_path "/media/djape/EXData/Grounding/data/render" \
  --topn_mask 150 \
  --topn_clip 100 \
  --topn_feature 10





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

python RANN_predict.py \
  -s /media/djape/EXData/Grounding/data \
  -m /media/djape/EXData/Grounding/data/output \
  --pointcloud_path /media/djape/EXData/Grounding/data/results/intersection.npy

python GSG_train.py -s /media/djape/EXData/Grounding/data -m /media/djape/EXData/Grounding/data/output \
  --checkpoint_path /media/djape/EXData/Grounding/data/output/chkpnt30000.pth \
  --intersection_path /media/djape/EXData/Grounding/data/results/intersection.npy \
  --threshold 0.0414