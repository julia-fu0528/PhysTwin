CAMERAS_BRICS="brics-odroid-001_cam0,brics-odroid-001_cam1,\
brics-odroid-002_cam0,\
brics-odroid-004_cam0,\
brics-odroid-006_cam0,\
brics-odroid-007_cam0,brics-odroid-007_cam1,\
brics-odroid-008_cam0,brics-odroid-008_cam1,\
brics-odroid-009_cam0,brics-odroid-009_cam1,\
brics-odroid-010_cam0,brics-odroid-010_cam1,\
brics-odroid-011_cam0,\
brics-odroid-012_cam0,brics-odroid-012_cam1,\
brics-odroid-013_cam0,brics-odroid-013_cam1,\
brics-odroid-014_cam0,brics-odroid-014_cam1,\
brics-odroid-015_cam0,brics-odroid-015_cam1,\
brics-odroid-016_cam0,\
brics-odroid-017_cam0,brics-odroid-017_cam1,\
brics-odroid-018_cam0,brics-odroid-018_cam1,\
brics-odroid-019_cam0,brics-odroid-019_cam1,\
brics-odroid-021_cam0,brics-odroid-021_cam1,\
brics-odroid-022_cam0,brics-odroid-022_cam1,\
brics-odroid-023_cam0,\
brics-odroid-024_cam0,brics-odroid-024_cam1,\
brics-odroid-025_cam0,brics-odroid-025_cam1,\
brics-odroid-027_cam0,brics-odroid-027_cam1,\
brics-odroid-028_cam0"


python utils/aruco_multiview.py \
    -w 5 \
    --height 7 \
    -l 0.02 \
    -s 0.005 \
    --image-dir /users/wfu16/data/users/wfu16/datasets/2025-10-23_snapshot_julia_aruco \
    --cameras ${CAMERAS_BRICS} \
    --camera-matrices /users/wfu16/data/users/wfu16/datasets/2025-10-14_julia_umi/calibration/intrinsics.npy \
    --dist-coeffs /users/wfu16/data/users/wfu16/datasets/2025-10-14_julia_umi/calibration/dist.npy \
    --output-dir /users/wfu16/data/users/wfu16/datasets/2025-10-23_snapshot_julia_aruco/aruco_results \
    -rs -r