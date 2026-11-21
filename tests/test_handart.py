import os
from pathlib import Path
from time import monotonic
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import torch

from handart.window import MainWindow
from handart.camera import CameraFrame
from handart.canvas import Canvas, HEIGHT, WIDTH
from handart.gestures import COLOR, DRAW, ERASE, RESIZE, GestureFilter, Recognition, PinchSizeController, matches_pose
from handart.model import GestureModel, LABELS, prepare_frame
from handart.tracking import HandTracker, TrackedHand

from PySide6.QtCore import QEventLoop, QTemporaryDir, QTimer
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication


cv2.setNumThreads(1)
PHOTO = Path(__file__).resolve().parent / "fixtures" / "images.jpg"


def hand_pose(label):
    points = np.zeros((21, 3))
    for mcp, x in zip((5, 9, 13, 17), (-0.42, 0, 0.32, 0.57)):
        straight = label == ERASE or label in (DRAW, COLOR) and (
            mcp == 5 or label == COLOR and mcp == 9
        )
        points[mcp:mcp + 4] = (
            [(x, 1, 0), (x, 1.4, 0), (x, 1.6, 0), (x, 1.8, 0)] if straight
            else [(x, 1, 0), (x, 1.25, 0), (x, 1.25, 0.22), (x, 1.12, 0.26)]
        )
    points[4] = points[8] + (0.21, 0, 0) if label == RESIZE else (-0.85, 0.6, 0)
    return points


def scores(winner, score=0.91):
    return {label: score if label == winner else (1 - score) / 3 for label in LABELS}


class RecognitionTests(unittest.TestCase):
    def test_each_pose_is_scale_and_translation_invariant(self):
        for label in LABELS:
            with self.subTest(label=label):
                points = hand_pose(label) * 0.08 + (0.3, 0.2, -0.1)
                self.assertEqual([name for name in LABELS if matches_pose(name, points, False)],
                                 [label])

    def test_switch_and_loss_require_new_stroke(self):
        recognition = GestureFilter(rescue=False)
        draw_points = hand_pose(DRAW)
        self.assertIsNone(recognition.update(scores(DRAW), draw_points, 0.0).active)
        self.assertIsNone(recognition.update(scores(DRAW), draw_points, 0.05).active)
        drawing = recognition.update(scores(DRAW), draw_points, 0.10)
        self.assertEqual(drawing.active, DRAW)

        held = recognition.update(scores(DRAW, 0.55), draw_points, 0.15)
        self.assertEqual(held.activation_id, drawing.activation_id)
        conflict = recognition.update(scores(DRAW), hand_pose(COLOR), 0.26)
        self.assertEqual(conflict.raw, DRAW)
        self.assertIsNone(conflict.active)
        self.assertIsNone(recognition.update({}, None, 0.27).active)
        for timestamp in (0.30, 0.35):
            self.assertIsNone(recognition.update(scores(ERASE), hand_pose(ERASE), timestamp).active)
        erased = recognition.update(scores(ERASE), hand_pose(ERASE), 0.40)
        self.assertEqual(erased.active, ERASE)
        self.assertNotEqual(erased.activation_id, drawing.activation_id)
        self.assertIsNone(recognition.update(scores(ERASE), hand_pose(ERASE), 0.71).active)

    def test_pinch_hysteresis(self):
        points = hand_pose(RESIZE)
        points[4] = points[8] + (0.37, 0, 0)
        self.assertFalse(matches_pose(RESIZE, points, holding=False))
        self.assertTrue(matches_pose(RESIZE, points, holding=True))

    def test_color_requires_pose_release_to_rearm(self):
        recognition = GestureFilter(rescue=False)
        points = hand_pose(COLOR)
        for timestamp in (0.00, 0.05, 0.10):
            first = recognition.update(scores(COLOR), points, timestamp)
        self.assertEqual(first.active, COLOR)
        self.assertIsNone(recognition.update(scores(DRAW), points, 0.15).active)
        for timestamp in (0.20, 0.25, 0.30):
            held = recognition.update(scores(COLOR), points, timestamp)
        self.assertEqual(held.active, COLOR)
        self.assertEqual(held.activation_id, first.activation_id)
        for timestamp in (0.35, 0.45):
            recognition.update({}, None, timestamp)
        for timestamp in (0.50, 0.55, 0.60):
            second = recognition.update(scores(COLOR), points, timestamp)
        self.assertEqual(second.active, COLOR)
        self.assertNotEqual(second.activation_id, first.activation_id)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_input_matches_live_demo(self):
        rng = np.random.default_rng(8)
        bgr = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_LINEAR)
        expected = torch.from_numpy(resized.copy()).permute(2, 0, 1).float().div(255)
        expected = ((expected - 0.5) / 0.5).unsqueeze(0)
        torch.testing.assert_close(prepare_frame(rgb), expected, rtol=0, atol=0)

    def test_models_load_from_assets_without_camera(self):
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        result = GestureModel().predict(rgb)
        self.assertEqual(tuple(result), LABELS)
        self.assertAlmostEqual(sum(result.values()), 1.0, places=6)
        with HandTracker() as tracker:
            self.assertIsNone(tracker.detect(rgb, monotonic()))

    @unittest.skipUnless(PHOTO.is_file(), "Falta tests/fixtures/images.jpg")
    def test_models_run_on_repository_photo(self):
        bgr = cv2.imread(str(PHOTO))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with HandTracker() as tracker:
            hand = tracker.detect(rgb, monotonic())
        self.assertIsNotNone(hand)
        self.assertEqual(hand.world_points.shape, (21, 3))
        self.assertTrue(matches_pose(ERASE, hand.world_points, False))
        result = GestureModel().predict(rgb)
        self.assertEqual(tuple(result), LABELS)
        self.assertAlmostEqual(sum(result.values()), 1.0, places=6)


class CanvasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setQuitOnLastWindowClosed(False)

    def test_draw_erase_undo_import_and_export(self):
        canvas = Canvas()
        canvas.track((0.5, 0.5), DRAW, 1, 0.0)
        canvas.track((0.59, 0.5), DRAW, 1, 0.08)
        canvas.track(None, None, 0, 0.12)
        canvas.track((0.85, 0.5), DRAW, 2, 0.17)
        canvas.track((0.5, 0.5), ERASE, 3, 0.22)
        temporary = QTemporaryDir()
        path = str(Path(temporary.path()) / "dibujo.png")
        canvas.save_png(path)
        image = QImage(path)
        self.assertEqual((image.width(), image.height()), (WIDTH, HEIGHT))
        self.assertEqual(image.pixelColor(600, 450).name(), "#ffffff")
        self.assertEqual(image.pixelColor(680, 450).name(), canvas.color.name())
        self.assertEqual(image.pixelColor(900, 450).name(), "#ffffff")
        canvas.undo()
        canvas.save_png(path)
        self.assertEqual(QImage(path).pixelColor(600, 450).name(), canvas.color.name())
        canvas.open_image(path)
        canvas.track((0.5, 0.5), ERASE, 4, 0.27)
        canvas.save_png(path)
        self.assertEqual(QImage(path).pixelColor(600, 450).name(), canvas.color.name())
        canvas.clear()
        canvas.save_png(path)
        self.assertEqual(QImage(path).pixelColor(600, 450).name(), "#ffffff")

    def test_window_consumes_frame_and_pauses(self):
        class FakeWorker:
            def __init__(self, frame):
                self.frame = frame
                self.interrupted = False

            def take_latest(self):
                frame, self.frame = self.frame, None
                return frame

            def requestInterruption(self):
                self.interrupted = True

        window = MainWindow()
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        hand = TrackedHand(((0.5, 0.5),) * 21, hand_pose(DRAW))
        frame = CameraFrame(rgb, hand, Recognition(DRAW, 0.91, DRAW, "Confirmado", 1), monotonic())
        worker = FakeWorker(frame)
        window._worker = worker
        window._set_camera_state("Iniciando…")
        window._poll_camera()
        self.assertEqual(window.gesture_sidebar.active_gesture, DRAW)
        self.assertTrue(window._undo_button.isEnabled())
        window._pause.click()
        self.assertTrue(worker.interrupted)
        self.assertIsNone(window.gesture_sidebar.active_gesture)
        window._worker = None
        window.close()

    def test_gestures_cycle_color_and_resize_shared_width(self):
        window = MainWindow()
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)

        captured_at = 0.0

        def gesture(label, activation, aperture=None):
            nonlocal captured_at
            captured_at += 0.05
            hand = TrackedHand(((0.5, 0.5),) * 21, hand_pose(label))
            result = Recognition(label, 0.91, label, "Confirmado", activation, aperture)
            window._apply_gesture(CameraFrame(rgb, hand, result, captured_at))

        gesture(COLOR, 1)
        first_color = window.canvas.color.name()
        self.assertEqual(first_color, "#088c7b")
        gesture(COLOR, 1)
        self.assertEqual(window.canvas.color.name(), first_color)
        gesture(COLOR, 2)
        self.assertEqual(window.canvas.color.name(), "#3468d9")

        gesture(RESIZE, 3, 0.25)
        self.assertEqual(window.canvas.width_px, 9)
        for _ in range(20):
            gesture(RESIZE, 3, 0.35)
        brush_width = window.canvas.width_px
        self.assertGreater(brush_width, 9)
        self.assertFalse(window._undo_button.isEnabled())

        gesture(ERASE, 4)
        self.assertEqual(window._tool_label.text(), "Borrador")
        self.assertEqual(window.canvas.width_px, brush_width)
        gesture(RESIZE, 5, 0.25)
        for _ in range(20):
            gesture(RESIZE, 5, 0.15)
        shared_width = window.canvas.width_px
        self.assertLess(shared_width, brush_width)
        self.assertEqual(window._width_slider.value(), window.canvas.width_px)
        gesture(DRAW, 6)
        self.assertEqual(window._tool_label.text(), "Pincel")
        self.assertEqual(window.canvas.width_px, shared_width)
        window._width_slider.setValue(65)
        gesture(ERASE, 7)
        self.assertEqual(window.canvas.width_px, 65)
        self.assertEqual(window._width_slider.value(), 65)
        window.close()

    def test_pinch_landmarks_resize_both_tools_without_painting(self):
        window = MainWindow()
        window._set_manual_width(30)
        recognition = GestureFilter()
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        observed = []
        apertures = [0.20] * 5 + [0.35] * 30 + [0.08] * 60
        for frame, aperture in enumerate(apertures):
            timestamp = frame / 30
            points = hand_pose(RESIZE)
            points[4] = points[8] + (aperture, 0, 0)
            result = recognition.update(scores(RESIZE), points, timestamp)
            hand = TrackedHand(((0.5, 0.5),) * 21, points)
            window._apply_gesture(CameraFrame(rgb, hand, result, timestamp))
            observed.append(window.canvas.width_px)
        self.assertGreater(max(observed), 30)
        self.assertLess(observed[-1], 30)
        self.assertTrue(all(abs(a - b) <= 1 for a, b in zip(observed, observed[1:])))
        self.assertFalse(window._undo_button.isEnabled())
        window.canvas.tool = ERASE
        window._sync_width()
        self.assertEqual(window.canvas.width_px, observed[-1])
        self.assertEqual(window._width_slider.value(), observed[-1])
        window.close()

    def test_worker_processes_frames_and_releases_capture(self):
        class PhotoCapture:
            def __init__(self, frame):
                self.frame = frame
                self.released = False

            def read(self):
                return True, self.frame

            def release(self):
                self.released = True

        def wait_for(condition):
            deadline = monotonic() + 5
            loop = QEventLoop()
            timer = QTimer(interval=10)

            def check():
                if condition() or monotonic() >= deadline:
                    loop.quit()

            timer.timeout.connect(check)
            timer.start()
            loop.exec()
            timer.stop()
            self.assertTrue(condition())

        photo = np.zeros((480, 640, 3), dtype=np.uint8)
        window = MainWindow()
        window.show()
        for stop in (window._stop_camera, window.close):
            capture = PhotoCapture(photo)
            with patch("handart.camera.open_camera", return_value=capture):
                window._start_camera()
                try:
                    wait_for(lambda: not window.preview._image.isNull())
                    self.assertEqual(window._camera_state, "En directo")
                    self.assertIsNone(window.gesture_sidebar.active_gesture)
                finally:
                    stop()
                    wait_for(lambda: window._worker is None)
            self.assertTrue(capture.released)


class PinchSizeTests(unittest.TestCase):
    def test_opening_then_closing_reduces_below_initial_width(self):
        controller = PinchSizeController((2, 96))
        width = controller.update(0.20, 0.0, 1, 30)
        for frame in range(1, 31):
            width = controller.update(0.35, frame / 30, 1, width)
        peak = width
        self.assertGreater(peak, 30)
        for frame in range(31, 91):
            width = controller.update(0.08, frame / 30, 1, width)
        self.assertLess(width, 30)

    def test_noise_and_step_are_smoothed_at_different_frame_rates(self):
        final_widths = []
        for fps in (15, 30, 60):
            controller = PinchSizeController((2, 96))
            width = controller.update(0.20, 0.0, 1, 30)
            for frame in range(1, fps + 1):
                aperture = 0.20 + (0.005 if frame % 2 else -0.005)
                width = controller.update(aperture, frame / fps, 1, width)
                self.assertEqual(width, 30)
            for frame in range(fps + 1, 3 * fps + 1):
                previous = width
                width = controller.update(0.40, frame / fps, 1, width)
                self.assertLessEqual(abs(width - previous), 1)
            final_widths.append(width)
        self.assertLessEqual(max(final_widths) - min(final_widths), 1)

    def test_new_pinch_tracking_gap_and_limits_do_not_jump(self):
        controller = PinchSizeController((2, 96))
        width = controller.update(0.20, 0.0, 1, 95)
        for frame in range(1, 31):
            width = controller.update(0.40, frame / 30, 1, width)
        self.assertEqual(width, 96)
        for frame in range(31, 61):
            width = controller.update(0.20, frame / 30, 1, width)
        self.assertLess(width, 96)
        self.assertEqual(controller.update(0.05, 2.05, 2, width), width)
        self.assertEqual(controller.update(0.40, 3.00, 2, width), width)
        self.assertEqual(controller.update(None, 3.05, 2, width), width)
        self.assertEqual(controller.update(0.20, 3.10, 2, width), width)
        self.assertEqual(controller.update(0.10, 3.15, 2, 2), 2)
