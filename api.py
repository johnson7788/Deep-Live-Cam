#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2024/8/24 15:28
# @File  : api.py
# @Author: Johnson
# @Desc  :  API，需要安装onnxruntime-gpu==1.18.0，使用cuda
import os
import sys
import logging
# single thread doubles cuda performance - needs to be set before torch import
# if any(arg.startswith('--execution-provider') for arg in sys.argv):
os.environ['OMP_NUM_THREADS'] = '1'
# reduce tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import shutil
import warnings
from flask import Flask, request, jsonify, send_file
import modules.globals
import modules.metadata
import modules.ui as ui
from modules.core import encode_execution_providers, decode_execution_providers, suggest_max_memory, suggest_execution_providers, suggest_execution_threads, \
    limit_resources, release_resources, pre_check, update_status,destroy
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import has_image_extension, is_image, is_video, detect_fps, create_video, extract_frames, \
    get_temp_frame_paths, restore_audio, create_temp, move_temp, clean_temp, normalize_output_path

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4'}

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')

class LiveCam():
    def __init__(self):
        self.upload_dir = "uploads"
        self.output_dir = "outputs"
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        modules.globals.headless = True
        modules.globals.execution_providers = decode_execution_providers(["cuda"])

    def setup_modules_globals(self, key_values_dict):
        """设置modules.globals的各种属性"""
        for key, value in key_values_dict.items():
            setattr(modules.globals, key, value)

    def process_image_to_image(self, source_path, target_path, output_path, options):
        output_path = normalize_output_path(source_path, target_path, output_path)
        self.setup_modules_globals(key_values_dict={
            'source_path': source_path,
            'target_path': target_path,
            'output_path': output_path,
            'frame_processors': options.get('frame_processor', ['face_swapper']),
            'keep_fps': options.get('keep_fps', False),
            'keep_audio': options.get('keep_audio', True),
            'keep_frames': options.get('keep_frames', False),
            'many_faces': options.get('many_faces', False),
            'nsfw_filter': options.get('nsfw_filter', False),
            'max_memory': options.get('max_memory', suggest_max_memory()),
            'execution_threads': options.get('execution_threads', suggest_execution_threads())
        })
        update_status('Processing image to image...')
        if modules.globals.nsfw_filter and ui.check_and_ignore_nsfw(modules.globals.target_path, destroy):
            return False
        try:
            shutil.copy2(modules.globals.target_path, modules.globals.output_path)
        except Exception as e:
            print("Error copying file:", str(e))
            return False
        for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
            update_status('Processing...', frame_processor.NAME)
            frame_processor.process_image(modules.globals.source_path, modules.globals.output_path, modules.globals.output_path)
            release_resources()
        if is_image(modules.globals.target_path):
            update_status('Processing to image succeeded!')
            return True
        else:
            update_status('Processing to image failed!')
            return False

    def process_image_to_video(self, source_path, target_path, output_path, options):
        # 看是否给出文件路径还是文件目录
        output_path = normalize_output_path(source_path, target_path, output_path)
        self.setup_modules_globals(key_values_dict={
            'source_path': source_path,
            'target_path': target_path,
            'output_path': output_path,
            'frame_processors': options.get('frame_processor', ['face_swapper']),
            'keep_fps': options.get('keep_fps', False),
            'keep_audio': options.get('keep_audio', True),
            'keep_frames': options.get('keep_frames', False),
            'many_faces': options.get('many_faces', False),
            'nsfw_filter': options.get('nsfw_filter', False),
            'video_encoder': options.get('video_encoder', 'libx264'),
            'video_quality': options.get('video_quality', 18),
            'max_memory': options.get('max_memory', suggest_max_memory()),
            'execution_threads': options.get('execution_threads', suggest_execution_threads())
        })
        update_status('Processing image to video...')
        if modules.globals.nsfw_filter and ui.check_and_ignore_nsfw(modules.globals.target_path, destroy):
            return False
        create_temp(modules.globals.target_path)
        extract_frames(modules.globals.target_path)
        temp_frame_paths = get_temp_frame_paths(modules.globals.target_path)
        for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
            update_status('Processing...', frame_processor.NAME)
            frame_processor.process_video(modules.globals.source_path, temp_frame_paths)
            release_resources()
        if modules.globals.keep_fps:
            fps = detect_fps(modules.globals.target_path)
            update_status(f'Creating video with {fps} fps...')
            create_video(modules.globals.target_path, fps)
        else:
            update_status('Creating video with 30.0 fps...')
            create_video(modules.globals.target_path)
        if modules.globals.keep_audio:
            restore_audio(modules.globals.target_path, modules.globals.output_path)
        else:
            move_temp(modules.globals.target_path, modules.globals.output_path)
        clean_temp(modules.globals.target_path)
        if is_video(modules.globals.target_path):
            update_status('Processing to video succeeded!')
            return True
        else:
            update_status('Processing to video failed!')
            return False


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/api/upload_file", methods=['POST'])
def upload_file_api():
    """
    上传一些数据
    """
    if 'file' not in request.files:
        return jsonify({"code": 4002, 'msg': 'No file in data', "data": None}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"code": 4003, 'msg': 'file Name is empty', "data": None}), 400
    logging.info(f"接收到上传文件请求,收到的文件名为：{file.filename}")
    if file:
        filename = file.filename.replace(" ", "_")
        file_path = os.path.join(livecam_instance.upload_dir, filename)
        file.save(file_path)
        file_path = file_path.split("uploads/", 1)[1]
        return jsonify({"code": 0, "msg": "success", "data": {'filePath': file_path, "filename": filename}}), 200
    return jsonify({"code": 4001, "msg": "文件不存在或者文件类型不符合", "data": None}), 4001

@app.route('/api/image_to_image', methods=['POST'])
def image_to_image():
    data = request.get_json()
    if not data or 'source_filename' not in data or 'target_filename' not in data:
        return jsonify({'error': 'Source and target filenames are required'}), 400

    source_filename = data['source_filename']
    target_filename = data['target_filename']

    if not (allowed_file(source_filename) and allowed_file(target_filename)):
        return jsonify({'error': 'Invalid file format'}), 400

    source_path = os.path.join(livecam_instance.upload_dir, source_filename)
    target_path = os.path.join(livecam_instance.upload_dir, target_filename)
    output_path = os.path.join(livecam_instance.output_dir, 'output_image.png')

    if not os.path.exists(source_path) or not os.path.exists(target_path):
        return jsonify({'error': 'File not found'}), 404

    result = livecam_instance.process_image_to_image(source_path, target_path, output_path, data)

    if result:
        return send_file(output_path, mimetype='image/png')
    else:
        return jsonify({'error': 'Image processing failed'}), 500

@app.route('/api/image_to_video', methods=['POST'])
def image_to_video():
    data = request.get_json()
    if not data or 'source_filename' not in data or 'target_filename' not in data:
        return jsonify({'error': 'Source image and target video filenames are required'}), 400

    source_filename = data['source_filename']
    target_filename = data['target_filename']

    if not (allowed_file(source_filename) and allowed_file(target_filename)):
        return jsonify({'error': 'Invalid file format'}), 400

    source_path = os.path.join(livecam_instance.upload_dir, source_filename)
    target_path = os.path.join(livecam_instance.upload_dir, target_filename)
    if not os.path.exists(source_path) or not os.path.exists(target_path):
        return jsonify({'error': 'source_path 或者 target_path not found'}), 404
    output_path = os.path.join(livecam_instance.output_dir, 'output_video.mp4')

    result = livecam_instance.process_image_to_video(source_path, target_path, output_path, data)

    if result:
        return send_file(output_path, mimetype='video/mp4')
    else:
        return jsonify({'error': 'Video processing failed'}), 500

if __name__ == '__main__':
    livecam_instance = LiveCam()
    app.run(host='0.0.0.0', port=6393)
