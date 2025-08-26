import cv2
import numpy as np
import os

output_dir = "extracted_cells"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)

binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


binary = cv2.bitwise_not(binary)

horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)


horizontal_segments = cv2.HoughLinesP(horizontal_lines, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
vertical_segments = cv2.HoughLinesP(vertical_lines, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)

h_lines = [] 
v_lines = [] 

if horizontal_segments is not None:
    for line in horizontal_segments:
        x1, y1, x2, y2 = line[0]
        h_lines.append(y1)


if vertical_segments is not None:
    for line in vertical_segments:
        x1, y1, x2, y2 = line[0]
        v_lines.append(x1)


h_lines = sorted(list(set(h_lines)))
v_lines = sorted(list(set(v_lines)))

if v_lines and v_lines[0] > 10:  
    v_lines.insert(0, 0)
if v_lines and v_lines[-1] < img.shape[1] - 10: 
    v_lines.append(img.shape[1])

if h_lines and h_lines[0] > 10:  
    h_lines.insert(0, 0)
if h_lines and h_lines[-1] < img.shape[0] - 10:  
    h_lines.append(img.shape[0])

print(f"Detected {len(h_lines)} horizontal lines and {len(v_lines)} vertical lines")
print(f"Horizontal lines at: {h_lines}")
print(f"Vertical lines at: {v_lines}")

cell_count = 0
extracted_cells = []

padding = 2  

for i in range(len(h_lines) - 1):
    for j in range(len(v_lines) - 1):
        # Define cell boundaries
        x1, y1 = v_lines[j], h_lines[i]
        x2, y2 = v_lines[j + 1], h_lines[i + 1]
        
        x1_pad = max(0, x1 + padding)
        y1_pad = max(0, y1 + padding)
        x2_pad = min(img.shape[1], x2 - padding)
        y2_pad = min(img.shape[0], y2 - padding)
        
        if (x2_pad - x1_pad) > 15 and (y2_pad - y1_pad) > 10:
            cell_img = img[y1_pad:y2_pad, x1_pad:x2_pad]
            
            if cell_img.shape[0] < 8 or cell_img.shape[1] < 8:
                continue
            
            row = i
            col = j
            
            filename = f"cell_r{row:02d}_c{col:02d}.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, cell_img)
            
            extracted_cells.append({
                'image': cell_img,
                'position': (row, col),
                'bbox': (x1_pad, y1_pad, x2_pad - x1_pad, y2_pad - y1_pad),
                'filename': filename
            })
            
            cell_count += 1

print(f"Extracted {cell_count} individual cell images to '{output_dir}' folder")

result_img = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)

for i, cell_data in enumerate(extracted_cells):
    x, y, w, h = cell_data['bbox']
    row, col = cell_data['position']
    
    cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    label = f"R{row}C{col}"
    cv2.putText(result_img, label, (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

cv2.imshow("Extracted Cells", result_img)
cv2.imwrite("extraction_visualization.png", result_img)


print("\nExtracted files:")
for cell_data in extracted_cells[:10]:  
    print(f"  {cell_data['filename']} - Row {cell_data['position'][0]}, Col {cell_data['position'][1]}")

if len(extracted_cells) > 10:
    print(f"  ... and {len(extracted_cells) - 10} more files")

print(f"\nAll cell images saved in '{output_dir}' folder")

cv2.waitKey(0)
cv2.destroyAllWindows()