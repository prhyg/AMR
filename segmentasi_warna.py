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

CONTROL_WINDOW = "Kontrol HSV"
HUE_MAX = 179
WIDTH_MAX = 89


def noop(_value):
    pass


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


def draw_objects(frame, contours, label):
    color = COLORS[label]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = int(cv2.contourArea(contour))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            frame,
            f"{label} {area}px",
            (x, max(y - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


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
        f"Merah: {len(red_objects)} objek",
        f"Biru : {len(blue_objects)} objek",
        f"Cakupan: {ratio:.1f}%",
        midpoint_info,
        "q/ESC keluar | s simpan | m mask | p simpan HSV",
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


def create_controls(config):
    cv2.namedWindow(CONTROL_WINDOW, cv2.WINDOW_NORMAL)
    for prefix, key in (("Merah", "merah"), ("Biru", "biru")):
        values = config[key]
        cv2.createTrackbar(f"{prefix} H tengah", CONTROL_WINDOW, values["center"], HUE_MAX, noop)
        cv2.createTrackbar(f"{prefix} H lebar", CONTROL_WINDOW, values["width"], WIDTH_MAX, noop)
        cv2.createTrackbar(f"{prefix} S min", CONTROL_WINDOW, values["smin"], 255, noop)
        cv2.createTrackbar(f"{prefix} V min", CONTROL_WINDOW, values["vmin"], 255, noop)


def read_controls():
    config = {}
    for prefix, key in (("Merah", "merah"), ("Biru", "biru")):
        config[key] = {
            "center": cv2.getTrackbarPos(f"{prefix} H tengah", CONTROL_WINDOW),
            "width": cv2.getTrackbarPos(f"{prefix} H lebar", CONTROL_WINDOW),
            "smin": cv2.getTrackbarPos(f"{prefix} S min", CONTROL_WINDOW),
            "vmin": cv2.getTrackbarPos(f"{prefix} V min", CONTROL_WINDOW),
        }
    return config


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

    cap = cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened():
        raise SystemExit(f"Gagal membuka sumber: {args.source}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    config = load_config(args.config)
    create_controls(config)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    window = "Segmentasi Merah & Biru"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    show_mask = False
    last_time = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if not args.no_flip:
            frame = cv2.flip(frame, 1)

        config = read_controls()
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

        display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if show_mask else result
        cv2.imshow(window, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("s"):
            filename = time.strftime("snapshot_%Y%m%d_%H%M%S.png")
            cv2.imwrite(filename, result)
            print(f"Disimpan: {filename}")
        if key == ord("m"):
            show_mask = not show_mask
        if key == ord("p"):
            save_config(args.config, config)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
