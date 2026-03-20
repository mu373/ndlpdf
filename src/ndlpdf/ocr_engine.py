"""Wrapper around ndlocr-lite OCR pipeline for per-page text extraction."""

from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import ocr


def load_models(device: str = "cpu"):
    """Load ndlocr-lite detection and recognition models.

    Model weights are resolved from the installed ndlocr-lite package.
    Three recognizers handle different character counts via cascade:
    rec30 (<=30 chars), rec50 (<=50), rec100 (<=100).

    Parameters
    ----------
    device : str
        Inference device, ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    tuple
        ``(detector, rec30, rec50, rec100)`` model objects.
    """
    base_dir = Path(ocr.__file__).resolve().parent
    args = argparse.Namespace(
        det_weights=str(base_dir / "model" / "deim-s-1024x1024.onnx"),
        det_classes=str(base_dir / "config" / "ndl.yaml"),
        rec_weights=str(
            base_dir / "model" / "parseq-ndl-16x768-100-tiny-165epoch-tegaki2.onnx"
        ),
        rec_weights30=str(
            base_dir / "model" / "parseq-ndl-16x256-30-tiny-192epoch-tegaki3.onnx"
        ),
        rec_weights50=str(
            base_dir / "model" / "parseq-ndl-16x384-50-tiny-146epoch-tegaki2.onnx"
        ),
        rec_classes=str(base_dir / "config" / "NDLmoji.yaml"),
        det_score_threshold=0.2,
        det_conf_threshold=0.25,
        det_iou_threshold=0.2,
        device=device,
    )
    detector = ocr.get_detector(args)
    rec30 = ocr.get_recognizer(args, weights_path=args.rec_weights30)
    rec50 = ocr.get_recognizer(args, weights_path=args.rec_weights50)
    rec100 = ocr.get_recognizer(args, weights_path=args.rec_weights)
    return detector, rec30, rec50, rec100


def ocr_page(np_image, imgname, detector, rec30, rec50, rec100) -> list[dict]:
    """Run OCR on a single page image.

    Replicates the ndlocr-lite pipeline: detection, reading-order sorting
    (XY-cut), and cascade text recognition.

    Parameters
    ----------
    np_image : np.ndarray
        Page image as ``(H, W, 3)`` RGB numpy array.
    imgname : str
        Identifier for the page (used in XML intermediate).
    detector : DEIM
        Detection model.
    rec30, rec50, rec100 : PARSEQ
        Recogniser models for the three-stage cascade.

    Returns
    -------
    list[dict]
        Each dict has keys ``bbox`` (``[x, y, w, h]`` in pixels),
        ``text``, ``is_vertical``, ``confidence``.
        Ordered by reading order.
    """
    from ndl_parser import convert_to_xml_string3
    from ocr import RecogLine, process_cascade, process_detector
    from reading_order.xy_cut.eval import eval_xml

    detections, classeslist = process_detector(
        detector,
        inputname=imgname,
        npimage=np_image,
        outputpath=os.devnull,
        issaveimg=False,
    )

    resultobj = [dict(), dict()]
    resultobj[0][0] = list()
    for i in range(17):
        resultobj[1][i] = []
    for det in detections:
        xmin, ymin, xmax, ymax = det["box"]
        conf = det["confidence"]
        char_count = det["pred_char_count"]
        if det["class_index"] == 0:
            resultobj[0][0].append([xmin, ymin, xmax, ymax])
        resultobj[1][det["class_index"]].append(
            [xmin, ymin, xmax, ymax, conf, char_count]
        )

    img_h, img_w = np_image.shape[:2]
    xmlstr = convert_to_xml_string3(img_w, img_h, imgname, classeslist, resultobj)
    xmlstr = "<OCRDATASET>" + xmlstr + "</OCRDATASET>"
    root = ET.fromstring(xmlstr)
    eval_xml(root, logger=None)

    alllineobj = []
    for idx, lineobj in enumerate(root.findall(".//LINE")):
        xmin = int(lineobj.get("X"))
        ymin = int(lineobj.get("Y"))
        line_w = int(lineobj.get("WIDTH"))
        line_h = int(lineobj.get("HEIGHT"))
        try:
            pred_char_cnt = float(lineobj.get("PRED_CHAR_CNT"))
        except (TypeError, ValueError):
            pred_char_cnt = 100.0

        # Clamp to image bounds
        xmin = max(0, min(xmin, img_w))
        ymin = max(0, min(ymin, img_h))
        x_end = max(0, min(xmin + line_w, img_w))
        y_end = max(0, min(ymin + line_h, img_h))
        if x_end <= xmin or y_end <= ymin:
            continue

        lineimg = np_image[ymin:y_end, xmin:x_end, :]
        alllineobj.append(RecogLine(lineimg, idx, pred_char_cnt))

    if len(alllineobj) == 0 and len(detections) > 0:
        page = root.find("PAGE")
        if page is None:
            raise ValueError("OCR XML did not include a PAGE element")
        for idx, det in enumerate(detections):
            xmin, ymin, xmax, ymax = det["box"]
            # Clamp to image bounds
            xmin = max(0, min(int(xmin), img_w))
            ymin = max(0, min(int(ymin), img_h))
            xmax = max(0, min(int(xmax), img_w))
            ymax = max(0, min(int(ymax), img_h))
            line_w = xmax - xmin
            line_h = ymax - ymin
            if line_w <= 0 or line_h <= 0:
                continue

            line_elem = ET.SubElement(page, "LINE")
            line_elem.set("TYPE", "本文")
            line_elem.set("X", str(xmin))
            line_elem.set("Y", str(ymin))
            line_elem.set("WIDTH", str(line_w))
            line_elem.set("HEIGHT", str(line_h))
            line_elem.set("CONF", f"{det['confidence']:0.3f}")
            pred_char_cnt = det.get("pred_char_count", 100.0)
            line_elem.set("PRED_CHAR_CNT", f"{pred_char_cnt:0.3f}")

            lineimg = np_image[ymin:ymax, xmin:xmax, :]
            alllineobj.append(RecogLine(lineimg, idx, pred_char_cnt))

    resultlinesall = process_cascade(
        alllineobj,
        rec30,
        rec50,
        rec100,
        is_cascade=True,
    )

    results = []
    for idx, lineobj in enumerate(root.findall(".//LINE")):
        lineobj.set("STRING", resultlinesall[idx])
        xmin = int(lineobj.get("X"))
        ymin = int(lineobj.get("Y"))
        line_w = int(lineobj.get("WIDTH"))
        line_h = int(lineobj.get("HEIGHT"))
        try:
            conf = float(lineobj.get("CONF"))
        except (TypeError, ValueError):
            conf = 0.0

        results.append(
            {
                "bbox": [xmin, ymin, line_w, line_h],
                "text": resultlinesall[idx],
                "is_vertical": line_h > line_w,
                "confidence": conf,
            }
        )

    return results
