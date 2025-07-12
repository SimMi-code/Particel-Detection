# yolo_square/labeling_ui.py

import os
import cv2

from .box_utils   import yolo_to_square, square_to_xyxy
from .io_helpers  import list_image_files, ensure_dir

MIN_SIZE = 10
MOVE_STEP = 1  # pixels to move with each arrow key press

def labeling_ui(
    image_folder="images_split",
    label_folder="yolo_dataset/labels"
):
    """
    A simple UI to review & correct YOLO‐style square boxes.

    • Left‐drag in empty space → draw new square  
    • Left‐click inside an existing box + drag → move that box  
    • 'c' → copy selected box  
    • 'v' → paste at mouse cursor  
    • '+' / '-' → grow/shrink selected box by 1px (up/right only)  
    • Arrow keys → move selected box by 1px  
    • 'd' → delete selected box  
    • 's' → save .txt AND update viz image  
    • 'n' → skip (no save)  
    • 'q' → quit completely
    """
    ensure_dir(label_folder)
    viz_folder = os.path.join(os.path.dirname(label_folder), "viz")
    ensure_dir(viz_folder)

    img_paths = list_image_files(image_folder)
    if not img_paths:
        print("❌ No images found in", image_folder)
        return

    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]
        base = os.path.splitext(os.path.basename(img_path))[0]
        lab  = os.path.join(label_folder, base + ".txt")

        # --- load existing labels into pixel coords ---
        boxes = []
        if os.path.exists(lab):
            with open(lab) as f:
                for line in f:
                    try:
                        cx, cy, side = yolo_to_square(line, img.shape)
                        x1,y1,x2,y2 = square_to_xyxy(cx, cy, side)
                        boxes.append([x1,y1,x2,y2])
                    except Exception:
                        pass

        sel        = -1
        drawing    = False
        moving     = False
        orig_box   = None
        p0         = (0,0)
        cursor     = (0,0)
        copied_box = None

        def clip_box(x1,y1,x2,y2):
            x1 = max(0, min(x1, W-1))
            y1 = max(0, min(y1, H-1))
            x2 = max(0, min(x2, W-1))
            y2 = max(0, min(y2, H-1))
            if x2-x1 < MIN_SIZE:
                x2 = x1 + MIN_SIZE
            if y2-y1 < MIN_SIZE:
                y2 = y1 + MIN_SIZE
            return [x1,y1,x2,y2]

        def on_mouse(evt, x, y, flags, param):
            nonlocal boxes, sel, drawing, moving, orig_box, p0, cursor
            cursor = (x,y)
            if evt == cv2.EVENT_LBUTTONDOWN:
                # did we click inside an existing box?
                for i,(x1,y1,x2,y2) in enumerate(boxes):
                    if x1<=x<=x2 and y1<=y<=y2:
                        sel = i
                        moving = True
                        orig_box = boxes[i].copy()
                        p0 = (x,y)
                        return
                # else start drawing a new box
                sel = -1
                drawing = True
                p0 = (x,y)

            elif evt == cv2.EVENT_MOUSEMOVE:
                if moving and sel != -1:
                    dx = x - p0[0]
                    dy = y - p0[1]
                    x1,y1,x2,y2 = orig_box
                    boxes[sel] = clip_box(x1+dx, y1+dy, x2+dx, y2+dy)

            elif evt == cv2.EVENT_LBUTTONUP:
                if drawing:
                    x1,y1 = p0
                    dx,dy = x - x1, y - y1
                    s = max(abs(dx),abs(dy))
                    x2 = x1 + (s if dx>=0 else -s)
                    y2 = y1 + (s if dy>=0 else -s)
                    x1,x2 = sorted([x1,x2]); y1,y2 = sorted([y1,y2])
                    if x2-x1 >= MIN_SIZE:
                        boxes.append([x1,y1,x2,y2])
                        sel = len(boxes)-1
                drawing = False
                moving  = False

        cv2.namedWindow(base, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(base, on_mouse)
        print(f"🖊️  Labeling: {base}")

        while True:
            disp = img.copy()
            # draw all boxes
            for i,(x1,y1,x2,y2) in enumerate(boxes):
                clr = (0,255,0) if i==sel else (255,0,0)
                cv2.rectangle(disp,(x1,y1),(x2,y2),clr,2)
            # preview new drawing
            if drawing:
                x1,y1 = p0; x2,y2 = cursor
                cv2.rectangle(disp,(x1,y1),(x2,y2),(200,200,200),1)

            cv2.imshow(base, disp)
            k = cv2.waitKey(20) & 0xFF

            if k == ord('q'):
                cv2.destroyAllWindows()
                return

            # save & viz
            if k == ord('s'):
                # 1) write labels
                with open(lab,"w") as f:
                    for x1,y1,x2,y2 in boxes:
                        side = x2 - x1
                        cx   = (x1 + x2)/2 / W
                        cy   = (y1 + y2)/2 / H
                        sn   = side / W
                        f.write(f"0 {cx:.6f} {cy:.6f} {sn:.6f} {sn:.6f}\n")
                # 2) update viz image
                viz = img.copy()
                for x1,y1,x2,y2 in boxes:
                    cv2.rectangle(viz,(x1,y1),(x2,y2),(255,0,0),2)
                viz_path = os.path.join(viz_folder, base + "_viz.jpg")
                cv2.imwrite(viz_path, viz)
                print(f"💾 Saved labels → {lab}")
                print(f"🖼️ Updated viz → {viz_path}")
                cv2.destroyWindow(base)
                break

            # skip
            if k == ord('n'):
                print(f"⏭️  Skipped {base}")
                cv2.destroyWindow(base)
                break

            # delete
            if k == ord('d') and sel != -1:
                boxes.pop(sel)
                print("🗑 Deleted box")
                sel = -1

            # copy / paste
            if k == ord('c') and sel != -1:
                copied_box = boxes[sel].copy()
                print("📋 Copied box")
            if k == ord('v') and copied_box is not None:
                x1,y1,x2,y2 = copied_box
                side = x2 - x1
                cx,cy = cursor
                half = side//2
                nx1,ny1 = cx-half, cy-half
                nx2,ny2 = nx1+side, ny1+side
                boxes.append(clip_box(nx1,ny1,nx2,ny2))
                sel = len(boxes)-1
                print("📌 Pasted box")

            # resize selected (+ / -)
            if sel != -1 and k in (ord('+'), ord('-')):
                x1,y1,x2,y2 = boxes[sel]
                if k == ord('+'):
                    # expand up & right
                    y1 -= 1; x2 += 1
                else:
                    # contract down & left
                    y1 += 1; x2 -= 1
                boxes[sel] = clip_box(x1,y1,x2,y2)

            # move with arrow keys
            if sel != -1 and k in (81,82,83,84):
                dx = -MOVE_STEP if k==81 else MOVE_STEP if k==83 else 0
                dy = -MOVE_STEP if k==82 else MOVE_STEP if k==84 else 0
                x1,y1,x2,y2 = boxes[sel]
                boxes[sel] = clip_box(x1+dx,y1+dy,x2+dx,y2+dy)

        # next image...

    cv2.destroyAllWindows()
    print("🎉 Done labeling.")
