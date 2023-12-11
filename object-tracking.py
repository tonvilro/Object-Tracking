import cv2
import numpy as np
import os

# Function to process a specific dataset folder
def process_folder(folder_name):
    base_path = "TLP/" + folder_name
    txt_file = os.path.join(base_path, "groundtruth_rect.txt")
    img_folder = os.path.join(base_path, "img")

    # Load initial bounding box data
    data = np.genfromtxt(txt_file, delimiter=',')
    frames = data[:, 0]  # Frame numbers
    bboxes = data[:, 1:5]  # Bounding boxes

    # Create a tracker instance
    tracker = cv2.TrackerKCF_create()

    # Initialize tracker with the first frame and bounding box
    frame_number = int(frames[0])
    filename = os.path.join(img_folder, f"{frame_number:05d}.JPG")
    frame = cv2.imread(filename)
    bbox = bboxes[0]

    # Convert numpy float type to integer for the bounding box
    ok = tracker.init(frame, bbox.astype(int))

    # Initialize video writer
    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_filename = os.path.join(base_path, "tracked_output.avi")
    out = cv2.VideoWriter(output_filename, fourcc, 20.0, (width, height))

    # Initialize a list to store updated bounding boxes
    updated_bboxes = []

    # Iterate through the rest of the frames
    for i in range(1, len(frames)):
        frame_number = int(frames[i])
        print(f"Processing frame {frame_number}/{len(frames)}")  # Progress update
        filename = os.path.join(img_folder, f"{frame_number:05d}.JPG")
        frame = cv2.imread(filename)

        # Update the tracker
        ok, bbox = tracker.update(frame)
        updated_bboxes.append(bbox)

        if ok:
            # Draw bounding box on the frame
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failed", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Write the frame into the file
        out.write(frame)

    # Release the video writer
    out.release()

    # Save the tracked data after processing all frames
    npz_filename = os.path.join(base_path, "tlp_tracked.npz")
    np.savez(npz_filename, frames=frames, bboxes=np.array(updated_bboxes))

# Main script
if __name__ == "__main__":
    folder_name = input("Enter the dataset folder name: ")
    process_folder(folder_name)
