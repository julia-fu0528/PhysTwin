import cv2
import numpy as np
from cv2 import aruco
import argparse
import time
import glob
import os

def main():
    parser = argparse.ArgumentParser(description='Detect ArUco GridBoard from multi-view images')
    parser.add_argument('-w', '--width', type=int, required=True, help='Number of markers in X direction')
    parser.add_argument('--height', type=int, required=True, help='Number of markers in Y direction')
    parser.add_argument('-l', '--length', type=float, required=True, help='Marker side length (in meters)')
    parser.add_argument('-s', '--separation', type=float, required=True, help='Separation between markers (in meters)')
    parser.add_argument('--image-dir', type=str, required=True, help='Directory containing camera folders')
    parser.add_argument('--cameras', nargs='+', default=['cam0', 'cam1', 'cam2'], help='Camera folder names')
    parser.add_argument('--camera-matrices', type=str, help='Path to camera matrices .npy file (N, 3, 3)')
    parser.add_argument('--dist-coeffs', type=str, help='Path to distortion coefficients .npy file (N, 5)')
    parser.add_argument('-d', '--dictionary', type=str, default='DICT_6X6_250', help='ArUco dictionary')
    parser.add_argument('--output-dir', type=str, default='aruco_output', help='Output directory for results')
    parser.add_argument('--image-pattern', type=str, default='*.jpg', help='Image file pattern')
    parser.add_argument('-r', '--show-rejected', action='store_true', help='Show rejected markers')
    parser.add_argument('-rs', '--refind-strategy', action='store_true', help='Use refind strategy')
    
    args = parser.parse_args()
    
    # Read parameters
    markers_x = args.width
    markers_y = args.height
    marker_length = args.length
    marker_separation = args.separation
    show_rejected = args.show_rejected
    refind_strategy = args.refind_strategy
    
    # Load camera calibration parameters
    if args.camera_matrices and args.dist_coeffs:
        cam_matrices = np.load(args.camera_matrices)  # Shape: (N_cameras, 3, 3)
        dist_coeffs_all = np.load(args.dist_coeffs)    # Shape: (N_cameras, 5) or (N_cameras, 5, 1)
        print(f"Loaded camera matrices: {cam_matrices.shape}")
        print(f"Loaded distortion coefficients: {dist_coeffs_all.shape}")
    else:
        print("Error: Camera calibration parameters required!")
        print("Use --camera-matrices and --dist-coeffs arguments")
        return
    
    # Ensure dist_coeffs is the right shape
    if len(dist_coeffs_all.shape) == 3:
        dist_coeffs_all = dist_coeffs_all.squeeze()
    
    # Select ArUco dictionary
    aruco_dict = aruco.getPredefinedDictionary(getattr(aruco, args.dictionary))
    detector_params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, detector_params)
    
    # Calculate axis length for visualization
    axis_length = 0.5 * (min(markers_x, markers_y) * (marker_length + marker_separation) + 
                         marker_separation)
    
    # Create GridBoard object
    board = aruco.GridBoard(
        (markers_x, markers_y),
        marker_length,
        marker_separation,
        aruco_dict
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Store results
    all_rvecs = {}
    all_tvecs = {}
    print(f"args.cameras: {args.cameras}")
    cameras = args.cameras[0].split(',')
    
    # Process each camera
    for cam_idx, camera_name in enumerate(cameras):
        print(f"\n{'='*60}")
        print(f"Processing camera: {camera_name}")
        print(f"{'='*60}")
        
        camera_dir = os.path.join(args.image_dir, camera_name)
        if not os.path.exists(camera_dir):
            print(f"Warning: Camera directory not found: {camera_dir}")
            continue
        
        # Get all images for this camera
        image_paths = sorted(glob.glob(os.path.join(camera_dir, args.image_pattern)))
        print(f"Found {len(image_paths)} images in {camera_dir}")
        
        if len(image_paths) == 0:
            print(f"No images found with pattern {args.image_pattern}")
            continue
        
        # Get camera calibration for this camera
        cam_matrix = cam_matrices[cam_idx]
        dist_coeffs = dist_coeffs_all[cam_idx].reshape(-1, 1)
        
        print(f"Camera matrix:\n{cam_matrix}")
        print(f"Distortion coefficients: {dist_coeffs.ravel()}")
        
        # Create output directory for this camera
        cam_output_dir = os.path.join(args.output_dir, camera_name)
        os.makedirs(cam_output_dir, exist_ok=True)
        
        rvecs_list = []
        tvecs_list = []
        
        # Process each image
        for img_idx, image_path in enumerate(image_paths):
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read image: {image_path}")
                continue
            
            image_copy = image.copy()
            
            # Detect markers
            corners, ids, rejected = detector.detectMarkers(image)
            
            # Refind strategy
            if refind_strategy and ids is not None:
                corners, ids, rejected, recovered_ids = detector.refineDetectedMarkers(
                    image, board, corners, ids, rejected, cam_matrix, dist_coeffs
                )
            
            # Estimate board pose
            rvec = None
            tvec = None
            markers_detected = 0
            
            if ids is not None and len(ids) > 0:
                obj_points, img_points = board.matchImagePoints(corners, ids)
                
                if obj_points is not None and len(obj_points) > 0:
                    success, rvec, tvec = cv2.solvePnP(
                        obj_points, img_points, cam_matrix, dist_coeffs
                    )
                    
                    if success:
                        markers_detected = len(obj_points) // 4
                        rvecs_list.append(rvec)
                        tvecs_list.append(tvec)
            
            # Draw results
            if ids is not None and len(ids) > 0:
                aruco.drawDetectedMarkers(image_copy, corners, ids)
            
            if show_rejected and rejected is not None and len(rejected) > 0:
                aruco.drawDetectedMarkers(image_copy, rejected, borderColor=(255, 0, 100))
            
            if rvec is not None and tvec is not None:
                cv2.drawFrameAxes(image_copy, cam_matrix, dist_coeffs, rvec, tvec, axis_length)
                
                # Display info
                cv2.putText(image_copy, f"Markers: {markers_detected}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(image_copy, f"Frame: {img_idx}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Save output image
            output_path = os.path.join(cam_output_dir, f"detected_{img_idx:06d}.png")
            cv2.imwrite(output_path, image_copy)
            
            # Optional: Display
            cv2.imshow(f"Detection - {camera_name}", image_copy)
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break
        
        cv2.destroyAllWindows()
        
        # Store results for this camera
        if len(rvecs_list) > 0:
            all_rvecs[camera_name] = np.array(rvecs_list)
            all_tvecs[camera_name] = np.array(tvecs_list)
            
            print(f"\nCamera {camera_name} - Detected board in {len(rvecs_list)}/{len(image_paths)} frames")
            
            # Compute average pose
            avg_rvec = np.mean(all_rvecs[camera_name], axis=0)
            avg_tvec = np.mean(all_tvecs[camera_name], axis=0)
            
            # Convert rotation vector to matrix
            R_avg, _ = cv2.Rodrigues(avg_rvec)
            
            print(f"Average rotation matrix:\n{R_avg}")
            print(f"Average translation: {avg_tvec.ravel()}")
            
            # Save results
            np.save(os.path.join(cam_output_dir, 'rvecs.npy'), all_rvecs[camera_name])
            np.save(os.path.join(cam_output_dir, 'tvecs.npy'), all_tvecs[camera_name])
            np.save(os.path.join(cam_output_dir, 'avg_rvec.npy'), avg_rvec)
            np.save(os.path.join(cam_output_dir, 'avg_tvec.npy'), avg_tvec)
            np.save(os.path.join(cam_output_dir, 'avg_rotation_matrix.npy'), R_avg)
    
    # Save all results
    print(f"all_rvecs: {all_rvecs}")
    print(f"all_tvecs: {all_tvecs}")
    np.save(os.path.join(args.output_dir, 'all_rvecs.npy'), all_rvecs)
    np.save(os.path.join(args.output_dir, 'all_tvecs.npy'), all_tvecs)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()