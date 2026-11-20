import argparse
import json
import math
import os
import time

import cv2
import numpy as np

COLORS = {
    "MERAH": (0, 0, 255),
    "BIRU": (255, 0, 0),
}

DEFAULTS = {
    "merah": {"center": 0, "width": 10, "smin": 100, "vmin": 50},
    "biru": {"center": 115, "width": 15, "smin": 100, "vmin": 50},
}

WINDOW = "Kamera & Tuning HSV"
HUE_MAX = 179
WIDTH_MAX = 89
TAB_HEIGHT = 36
TABS = ["Kamera", "Tuning HSV"]


def hsv_ranges(center, width, smin, vmin):
    low_h = (center - width) % 180
    high_h = (center + width) % 180
    if low_h <= high_h:
        return [((low_h, smin, vmin), (high_h, 255, 255))]
    return [
        ((low_h, smin, vmin), (179, 255, 255)),
        ((0, smin, vmin), (high_h, 255, 255)),
    ]


def build_mask(hsv, ranges):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask |= cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
    return mask


def clean_mask(mask, kernel):
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def find_objects(mask, min_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def select_nearest(contours, frame_center):
    if not contours:
        return []
    cx, cy = frame_center
    best = min(
        contours,
        key=lambda c: (contour_center(c)[0] - cx) ** 2 + (contour_center(c)[1] - cy) ** 2,
    )
    return [best]


def mask_from_contours(contours, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    if contours:
        cv2.drawContours(mask, contours, -1, 255, -1)
    return mask


def draw_objects(frame, contours, label):
    color = COLORS[label]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)


def contour_center(contour):
    moments = cv2.moments(contour)
    if moments["m00"] > 0:
        return int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])
    x, y, w, h = cv2.boundingRect(contour)
    return x + w // 2, y + h // 2


def find_midpoint(red_objects, blue_objects):
    objects = [(c, "MERAH") for c in red_objects] + [(c, "BIRU") for c in blue_objects]
    objects.sort(key=lambda item: cv2.contourArea(item[0]), reverse=True)
    if len(objects) < 2:
        return None
    first, second = objects[0], objects[1]
    p1 = contour_center(first[0])
    p2 = contour_center(second[0])
    midpoint = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
    distance = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
    return midpoint, p1, p2, distance


def draw_midpoint(frame, midpoint_data):
    if midpoint_data is None:
        return
    midpoint, p1, p2, distance = midpoint_data
    green = (0, 255, 0)
    cv2.line(frame, p1, p2, green, 1, cv2.LINE_AA)
    cv2.circle(frame, p1, 4, green, -1, cv2.LINE_AA)
    cv2.circle(frame, p2, 4, green, -1, cv2.LINE_AA)
    cv2.circle(frame, midpoint, 12, green, 2, cv2.LINE_AA)
    cv2.circle(frame, midpoint, 3, green, -1, cv2.LINE_AA)
    cv2.line(frame, (midpoint[0] - 18, midpoint[1]), (midpoint[0] + 18, midpoint[1]), green, 1, cv2.LINE_AA)
    cv2.line(frame, (midpoint[0], midpoint[1] - 18), (midpoint[0], midpoint[1] + 18), green, 1, cv2.LINE_AA)
    text = f"TITIK TENGAH ({midpoint[0]},{midpoint[1]}) jarak {distance:.0f}px"
    x = min(midpoint[0] + 16, frame.shape[1] - 360)
    y = max(midpoint[1] - 16, 24)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, green, 2, cv2.LINE_AA)


def process(frame, min_area, kernel, red_ranges, blue_ranges):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red_mask = clean_mask(build_mask(hsv, red_ranges), kernel)
    blue_mask = clean_mask(build_mask(hsv, blue_ranges), kernel)

    red_objects = find_objects(red_mask, min_area)
    blue_objects = find_objects(blue_mask, min_area)
    frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)
    red_objects = select_nearest(red_objects, frame_center)
    blue_objects = select_nearest(blue_objects, frame_center)
    red_mask = mask_from_contours(red_objects, frame.shape[:2])
    blue_mask = mask_from_contours(blue_objects, frame.shape[:2])

    overlay = frame.copy()
    overlay[red_mask > 0] = COLORS["MERAH"]
    overlay[blue_mask > 0] = COLORS["BIRU"]
    result = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

    draw_objects(result, red_objects, "MERAH")
    draw_objects(result, blue_objects, "BIRU")

    midpoint_data = find_midpoint(red_objects, blue_objects)
    draw_midpoint(result, midpoint_data)

    combined_mask = cv2.bitwise_or(red_mask, blue_mask)
    total_pixels = frame.shape[0] * frame.shape[1]
    ratio = cv2.countNonZero(combined_mask) / total_pixels * 100.0

    if midpoint_data is not None:
        midpoint = midpoint_data[0]
        midpoint_info = f"Titik tengah: ({midpoint[0]}, {midpoint[1]})"
    else:
        midpoint_info = "Titik tengah: butuh 2 objek"

    info = [
        f"Cakupan: {ratio:.1f}%",
        midpoint_info,
        "q/ESC keluar | s simpan | m/tab ganti | p simpan HSV",
    ]
    start_y = result.shape[0] - 12 - (len(info) - 1) * 26
    for i, text in enumerate(info):
        cv2.putText(
            result,
            text,
            (10, start_y + i * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return result, combined_mask


def load_config(path):
    config = json.loads(json.dumps(DEFAULTS))
    if path and os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            for key in ("merah", "biru"):
                if key in data:
                    config[key].update(
                        {k: int(v) for k, v in data[key].items() if k in config[key]}
                    )
            print(f"Konfigurasi HSV dimuat: {path}")
        except (OSError, ValueError, TypeError) as error:
            print(f"Gagal memuat {path}: {error}. Memakai nilai default.")
    return config


def save_config(path, config):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    print(f"Konfigurasi HSV disimpan: {path}")


SLIDER_DEFS = [
    ("Merah", "merah", [
        ("H tengah", "center", HUE_MAX),
        ("H lebar", "width", WIDTH_MAX),
        ("S min", "smin", 255),
        ("V min", "vmin", 255),
    ]),
    ("Biru", "biru", [
        ("H tengah", "center", HUE_MAX),
        ("H lebar", "width", WIDTH_MAX),
        ("S min", "smin", 255),
        ("V min", "vmin", 255),
    ]),
]


def probe_cameras(max_index=8):
    found = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        if cap is not None and cap.isOpened():
            found.append(i)
        if cap is not None:
            cap.release()
    return found


def draw_tuning_panel(width, height, config, scroll_y=0, cam_index=0, cameras=None):
    if cameras is None:
        cameras = []
    panel = np.full((height, width, 3), 35, dtype=np.uint8)
    pad = 20
    col_w = max((width - pad * 3) // 2, 220)
    rects = []
    cam_rects = []
    content_bottom = 0

    cam_y = 20 - scroll_y
    if -30 < cam_y < height:
        cv2.putText(panel, "Kamera:", (pad, cam_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
    bx = pad + 90
    cam_label = str(cam_index)
    for idx in cameras:
        label = str(idx)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        bw, bh = max(tw + 18, 36), 26
        rect = (bx, cam_y, bx + bw, cam_y + bh)
        cam_rects.append((rect, idx))
        if -30 < cam_y < height + 30:
            selected = idx == cam_index
            bg = (80, 160, 80) if selected else (70, 70, 70)
            border = (220, 255, 220) if selected else (140, 140, 140)
            cv2.rectangle(panel, (rect[0], rect[1]), (rect[2], rect[3]), bg, -1)
            cv2.rectangle(panel, (rect[0], rect[1]), (rect[2], rect[3]), border, 1)
            tx = rect[0] + (bw - tw) // 2
            ty = rect[1] + (bh + th) // 2 - 2
            cv2.putText(panel, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        bx += bw + 8
    if not cameras and -30 < cam_y < height:
        cv2.putText(panel, "(tidak terdeteksi)", (bx, cam_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    content_bottom = max(content_bottom, cam_y + 34 + scroll_y)

    col_start = cam_y + 44
    for col, (title, key, items) in enumerate(SLIDER_DEFS):
        x0 = pad + col * (col_w + pad)
        y = col_start
        title_color = COLORS["MERAH"] if key == "merah" else COLORS["BIRU"]
        if -30 < y < height:
            cv2.putText(panel, title, (x0, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.8, title_color, 2, cv2.LINE_AA)
        y += 30
        for label, field, maxv in items:
            val = int(config[key][field])
            track_x0 = x0 + 90
            track_x1 = x0 + col_w - 70
            if track_x1 < track_x0 + 40:
                track_x1 = track_x0 + 40
            track_y = y + 10
            rect = (track_x0, track_y - 8, track_x1, track_y + 8)
            rects.append((rect, key, field, maxv))
            if -20 < track_y < height + 20:
                cv2.putText(panel, label, (x0, track_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
                cv2.rectangle(panel, (track_x0, track_y - 3), (track_x1, track_y + 3), (90, 90, 90), -1)
                t = 0.0 if maxv == 0 else val / maxv
                kx = int(track_x0 + t * (track_x1 - track_x0))
                cv2.rectangle(panel, (track_x0, track_y - 3), (kx, track_y + 3), title_color, -1)
                cv2.circle(panel, (kx, track_y), 7, (240, 240, 240), -1)
                cv2.putText(panel, str(val), (track_x1 + 8, track_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y += 44
        sy = y + 10
        if -20 < sy < height + 20:
            cv2.putText(
                panel,
                f"center={config[key]['center']} width={config[key]['width']} s>={config[key]['smin']} v>={config[key]['vmin']}",
                (x0, sy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )
        content_bottom = max(content_bottom, y + 30 + scroll_y)
    hint = "Pilih kamera di atas | Scroll/drag untuk pan | p simpan | m/tab ganti | q keluar"
    cv2.putText(panel, hint, (pad, max(height - 20, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    max_scroll = max(0, content_bottom - (height - 40))
    if max_scroll > 0:
        bar_h = max(int(height * height / max(height + max_scroll, 1)), 30)
        bar_y = int((height - bar_h) * (scroll_y / max_scroll)) if max_scroll else 0
        cv2.rectangle(panel, (width - 8, bar_y), (width - 3, bar_y + bar_h), (120, 120, 120), -1)
    return panel, rects, cam_rects, max_scroll


def draw_tabs(frame, active_tab):
    h, w = frame.shape[:2]
    tab_w = w // len(TABS)
    bar = np.full((TAB_HEIGHT, w, 3), 45, dtype=np.uint8)
    for i, name in enumerate(TABS):
        x0 = i * tab_w
        x1 = w if i == len(TABS) - 1 else (i + 1) * tab_w
        color = (80, 160, 80) if i == active_tab else (70, 70, 70)
        cv2.rectangle(bar, (x0 + 1, 1), (x1 - 2, TAB_HEIGHT - 2), color, -1)
        cv2.rectangle(bar, (x0 + 1, 1), (x1 - 2, TAB_HEIGHT - 2), (200, 200, 200), 1)
        text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        tx = x0 + (x1 - x0 - text_size[0]) // 2
        ty = (TAB_HEIGHT + text_size[1]) // 2 - 2
        cv2.putText(bar, name, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, frame])


def make_tab_callback(state):
    def on_mouse(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and y < TAB_HEIGHT:
            tab_w = state["width"] // len(TABS)
            state["active_tab"] = min(x // max(tab_w, 1), len(TABS) - 1)
            state["dragging"] = None
            state["panning"] = None
            return
        if state.get("active_tab") != 1:
            return
        py = y - TAB_HEIGHT
        config = state["config"]
        if event == cv2.EVENT_LBUTTONDOWN:
            for rect, idx in state.get("cam_rects", []):
                x0, y0, x1, y1 = rect
                if x0 <= x <= x1 and y0 <= py <= y1:
                    state["requested_source"] = idx
                    state["dragging"] = None
                    state["panning"] = None
                    return
            for rect, key, field, maxv in state.get("slider_rects", []):
                x0, y0, x1, y1 = rect
                if x0 - 8 <= x <= x1 + 8 and y0 - 8 <= py <= y1 + 8:
                    state["dragging"] = (key, field, maxv, x0, x1)
                    val = round((x - x0) / max(x1 - x0, 1) * maxv)
                    config[key][field] = int(max(0, min(maxv, val)))
                    return
            state["panning"] = (y, state.get("scroll_y", 0))
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            if state.get("dragging"):
                key, field, maxv, x0, x1 = state["dragging"]
                val = round((x - x0) / max(x1 - x0, 1) * maxv)
                config[key][field] = int(max(0, min(maxv, val)))
            elif state.get("panning"):
                start_y, start_scroll = state["panning"]
                max_scroll = state.get("max_scroll", 0)
                state["scroll_y"] = int(max(0, min(max_scroll, start_scroll + (start_y - y))))
        elif event == cv2.EVENT_LBUTTONUP:
            state["dragging"] = None
            state["panning"] = None
        elif event == cv2.EVENT_MOUSEWHEEL:
            max_scroll = state.get("max_scroll", 0)
            step = 40
            delta = step if flags > 0 else -step
            state["scroll_y"] = int(max(0, min(max_scroll, state.get("scroll_y", 0) - delta)))
    return on_mouse


def parse_source(value):
    return int(value) if value.isdigit() else value


def main():
    parser = argparse.ArgumentParser(description="Segmentasi warna merah dan biru dari kamera")
    parser.add_argument("--source", default="0", help="indeks kamera atau path video")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--min-area", type=int, default=1000, help="luas minimum objek (piksel)")
    parser.add_argument("--no-flip", action="store_true", help="jangan flip gambar (mode cermin)")
    parser.add_argument("--config", default="hsv_config.json", help="file konfigurasi HSV")
    args = parser.parse_args()

    source = parse_source(args.source)
    cameras = probe_cameras()
    if isinstance(source, int) and source not in cameras:
        cameras = sorted(cameras + [source])
    cam_index = source if isinstance(source, int) else -1

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Gagal membuka sumber: {args.source}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    config = load_config(args.config)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(WINDOW, 1600, 940)
    tab_state = {
        "active_tab": 0,
        "width": max(args.width, 1),
        "config": config,
        "dragging": None,
        "panning": None,
        "slider_rects": [],
        "cam_rects": [],
        "scroll_y": 0,
        "max_scroll": 0,
        "cam_index": cam_index,
        "cameras": cameras,
        "requested_source": None,
    }
    cv2.setMouseCallback(WINDOW, make_tab_callback(tab_state))
    last_time = time.time()
    fps = 0.0

    while True:
        if tab_state.get("requested_source") is not None:
            new_src = tab_state["requested_source"]
            tab_state["requested_source"] = None
            if new_src != source:
                cap.release()
                cap = cv2.VideoCapture(new_src)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
                source = new_src
                tab_state["cam_index"] = new_src if isinstance(new_src, int) else -1

        ok, frame = cap.read()
        if not ok:
            if isinstance(source, str):
                break
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or args.height or 480)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or args.width or 640)
            frame = np.zeros((max(h, 1), max(w, 1), 3), dtype=np.uint8)
            cv2.putText(frame, f"Gagal membaca kamera: {source}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        if not args.no_flip:
            frame = cv2.flip(frame, 1)

        config = tab_state["config"]
        red_ranges = hsv_ranges(
            config["merah"]["center"],
            config["merah"]["width"],
            config["merah"]["smin"],
            config["merah"]["vmin"],
        )
        blue_ranges = hsv_ranges(
            config["biru"]["center"],
            config["biru"]["width"],
            config["biru"]["smin"],
            config["biru"]["vmin"],
        )

        result, mask = process(frame, args.min_area, kernel, red_ranges, blue_ranges)

        now = time.time()
        delta = now - last_time
        last_time = now
        if delta > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / delta)
        cv2.putText(
            result,
            f"FPS: {fps:.1f}",
            (result.shape[1] - 150, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        if tab_state["active_tab"] == 1:
            panel, rects, cam_rects, max_scroll = draw_tuning_panel(
                result.shape[1],
                result.shape[0],
                config,
                tab_state.get("scroll_y", 0),
                tab_state.get("cam_index", -1),
                tab_state.get("cameras", []),
            )
            tab_state["slider_rects"] = rects
            tab_state["cam_rects"] = cam_rects
            tab_state["max_scroll"] = max_scroll
            if tab_state.get("scroll_y", 0) > max_scroll:
                tab_state["scroll_y"] = max_scroll
            view = panel
        else:
            tab_state["slider_rects"] = []
            tab_state["cam_rects"] = []
            view = result
        tab_state["width"] = view.shape[1]
        cv2.imshow(WINDOW, draw_tabs(view, tab_state["active_tab"]))

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("s"):
            filename = time.strftime("snapshot_%Y%m%d_%H%M%S.png")
            cv2.imwrite(filename, result)
            print(f"Disimpan: {filename}")
        if key == ord("m"):
            tab_state["active_tab"] = 1 - tab_state["active_tab"]
        if key == ord("p"):
            save_config(args.config, config)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
