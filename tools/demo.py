#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
import shutil
import pandas as pd
import numpy as np
from loguru import logger

import cv2

import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
from yolox.utils.extra import classificar_movimento, preprocess_frame

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="../teste/", type=str, help="directory containing videos"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
            self,
            model,
            exp,
            cls_names=COCO_CLASSES,
            trt_file=None,
            decoder=None,
            device="cpu",
            fp16=False,
            legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        self.centroids = []
        self.person_centroids = []
        if trt_file is not None:
            from torch2trt import TRTModule # type: ignore

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True # True para filtrar uma classe, False para detectar multiplas classes
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35, timestamp=None):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img, None, None
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        df = self.make_df(bboxes, cls, timestamp, cls_conf, scores)
        df_person = self.make_df_person(bboxes, cls, timestamp, cls_conf, scores)

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res, df, df_person

    def make_df(self, bboxes, cls, timestamp, cls_conf, scores):
        for index in range(bboxes.size(0)):
            bbox_class = self.class_name(cls, index)
            
            # Inicializando as variáveis
            centroid_x = centroid_y = width = height = area = proportion_rate = None

            if bbox_class in ["head", "standing", "laying"] and scores[index] > cls_conf:
                centroid_x, centroid_y = self.centroid(bboxes[index])

                if bbox_class in ["standing", "laying"]:
                    width = self.width(bboxes[index])
                    height = self.height(bboxes[index])
                    area = self.area_bbox(bboxes[index])
                    proportion_rate = self.proportion_rate(bboxes[index])

                self.centroids.append((timestamp, bbox_class, centroid_x, centroid_y, width, height, area, proportion_rate))

        df = pd.DataFrame(self.centroids,
                        columns=['tempo', 'Classe', 'centroid_x', 'centroid_y', 'width', 'height', 'area', 'proportion'])

        df['status'] = df['Classe'].apply(lambda x: 'standing' if x == 'standing' else ('laying' if x == 'laying' else None))
        df['proportion'] = df['proportion'].round(2)
        df['area'] = df['area'].round(2)
        df['width'] = df['width'].round(2)
        df['height'] = df['height'].round(2)

        standing_df = df[df['Classe'] == 'standing'].copy()
        laying_df = df[df['Classe'] == 'laying'].copy()
        head_df = df[df['Classe'] == 'head'].copy()

        combined_df = pd.merge(standing_df, laying_df, on='tempo', how='outer')
        combined_df = pd.merge(combined_df, head_df, on='tempo', how='outer')

        combined_df = combined_df.rename(columns={
            'centroid_x_x': 'standing_x',
            'centroid_y_x': 'standing_y',
            'centroid_x_y': 'deitado_x',
            'centroid_y_y': 'deitado_y',
            'centroid_x': 'cabeca_x',
            'centroid_y': 'cabeca_y',
        })

        combined_df['standing_x'] = combined_df['standing_x'].fillna(combined_df['deitado_x'])
        combined_df['standing_y'] = combined_df['standing_y'].fillna(combined_df['deitado_y'])
        combined_df['deitado_x'] = combined_df['deitado_x'].fillna(combined_df['standing_x'])
        combined_df['deitado_y'] = combined_df['deitado_y'].fillna(combined_df['standing_y'])

        combined_df = combined_df.dropna(subset=['standing_x', 'standing_y', 'deitado_x', 'deitado_y'])

        standing_xy = [0]
        deitado_xy = [0]
        cabeca_xy = [0]

        for i in range(1, len(combined_df)):
            standing_ponto_atual = (combined_df['standing_x'].iloc[i], combined_df['standing_y'].iloc[i])
            standing_ponto_anterior = (combined_df['standing_x'].iloc[i - 1], combined_df['standing_y'].iloc[i - 1])
            standing_distancia = self.distancia_euclidiana(standing_ponto_atual, standing_ponto_anterior)
            standing_xy.append(standing_distancia)

            deitado_ponto_atual = (combined_df['deitado_x'].iloc[i], combined_df['deitado_y'].iloc[i])
            deitado_ponto_anterior = (combined_df['deitado_x'].iloc[i - 1], combined_df['deitado_y'].iloc[i - 1])
            deitado_distancia = self.distancia_euclidiana(deitado_ponto_atual, deitado_ponto_anterior)
            deitado_xy.append(deitado_distancia)

            cabeca_ponto_atual = (combined_df['cabeca_x'].iloc[i], combined_df['cabeca_y'].iloc[i])
            cabeca_ponto_anterior = (combined_df['cabeca_x'].iloc[i - 1], combined_df['cabeca_y'].iloc[i - 1])
            cabeca_distancia = self.distancia_euclidiana(cabeca_ponto_atual, cabeca_ponto_anterior)
            cabeca_xy.append(cabeca_distancia)

        combined_df['standing_xy'] = standing_xy
        combined_df['deitado_xy'] = deitado_xy
        combined_df['cabeca_xy'] = cabeca_xy

        combined_df['corpo_xy'] = combined_df['standing_xy'].fillna(combined_df['deitado_xy'])
        combined_df['tempo'] = combined_df['tempo'].round(decimals=2)
        combined_df = combined_df.drop_duplicates(subset=['tempo'], keep='first')
        combined_df['status'] = combined_df['status_x'].fillna(combined_df['status_y'])
        combined_df['proportion'] = combined_df['proportion_x'].fillna(combined_df['proportion_y'])
        combined_df['area'] = combined_df['area_x'].fillna(combined_df['area_y'])
        combined_df['width'] = combined_df['width_x'].fillna(combined_df['width_y'])
        combined_df['height'] = combined_df['height_x'].fillna(combined_df['height_y'])
        combined_df = combined_df.drop(
            columns=['status_x', 'status_y', 'area_x', 'area_y', 'proportion_x', 'proportion_y', 'width_x', 'width_y',
                    'height_x', 'height_y'])
        combined_df['corpo_xy'] = combined_df['corpo_xy'].round(2)
        combined_df['cabeca_xy'] = combined_df['cabeca_xy'].round(2)

        return combined_df
    
    def make_df_person(self, bboxes, cls, timestamp, cls_conf, scores):
        for index in range(bboxes.size(0)):
            bbox_class = self.class_name(cls, index)

            # Inicializando as variáveis
            centroid_x = centroid_y = None

            if bbox_class == "person" and scores[index] > cls_conf:
                centroid_x, centroid_y = self.centroid(bboxes[index])
                self.person_centroids.append((timestamp, bbox_class, centroid_x, centroid_y))

        df_person = pd.DataFrame(self.person_centroids,
                                columns=['tempo', 'Classe', 'centroid_x', 'centroid_y'])

        # Definir o status como "person" para todas as instâncias
        df_person['status'] = df_person['Classe'].apply(lambda x: 'person' if x == 'person' else None)

        # Arredondar os valores de centroid_x e centroid_y
        df_person['centroid_x'] = df_person['centroid_x'].round(2)
        df_person['centroid_y'] = df_person['centroid_y'].round(2)

        df_person['tempo'] = df_person['tempo'].round(decimals=2)
        df_person = df_person.drop_duplicates(subset=['tempo'], keep='first')

        return df_person


    def width(self, bounding_box):
        x_min = bounding_box[0].item()
        x_max = bounding_box[2].item()

        # Largura da bouding box
        width = x_max - x_min

        return width

    def height(self, bounding_box):
        y_min = bounding_box[1].item()
        y_max = bounding_box[3].item()

        # Altura da bouding box
        height = y_max - y_min

        return height

    def area_bbox(self, bounding_box):
        x_min = bounding_box[0].item()
        y_min = bounding_box[1].item()
        x_max = bounding_box[2].item()
        y_max = bounding_box[3].item()

        # Tamanho da bounding box
        box_area = (x_max - x_min) * (y_max - y_min)

        return box_area

    def proportion_rate(self, bounding_box):
        x_min = bounding_box[0].item()
        y_min = bounding_box[1].item()
        x_max = bounding_box[2].item()
        y_max = bounding_box[3].item()

        # Tamanho da bounding box
        box_area = (x_max - x_min) * (y_max - y_min)

        # Tamanho total da tela

        total_screen_area = 640 * 360
        # total_screen_area = 1280 * 720
        # total_screen_area = 1920 * 1080

        proportion_rate = box_area / total_screen_area

        return proportion_rate

    def distancia_euclidiana(self, ponto1, ponto2):
        return np.sqrt((ponto1[0] - ponto2[0]) ** 2 + (ponto1[1] - ponto2[1]) ** 2)

    def centroid(self, bounding_box):
        x_min = bounding_box[0].item()
        y_min = bounding_box[1].item()
        x_max = bounding_box[2].item()
        y_max = bounding_box[3].item()

        centroid_x = (x_min + x_max) / 2
        centroid_y = (y_min + y_max) / 2

        return centroid_x, centroid_y

    def class_name(self, cls, index):
        class_index = int(cls[index])
        class_name = self.cls_names[class_index]

        return class_name


def image_demo(predictor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    for image_name in files:
        outputs, img_info = predictor.inference(image_name)
        result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break


def get_video_path(video_dir):

    if os.path.isfile(video_dir):
        return video_dir

    files = os.listdir(video_dir)

    video_files = [f for f in files if f.lower().endswith('.mp4')]

    if video_files:
        return os.path.join(video_dir, video_files[0])
    else:
        return None


def imageflow_demo(predictor, vis_folder, current_time, args, intervalo=0.5):

    video_path = get_video_path(args.path)
    args.path = video_path
    if video_path:
        print(f"video utilizado: {video_path}")
    if video_path is None:
        print("Nenhum vídeo encontrado no diretório especificado.")
        return

    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = -1
    save_interval = int(fps * intervalo)
    if args.save_result:
        save_folder = os.path.join(
            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
        os.makedirs(save_folder, exist_ok=True)
        if args.demo == "video":
            save_path = os.path.join(save_folder, os.path.basename(args.path))
        else:
            save_path = os.path.join(save_folder, "camera.mp4")
        logger.info(f"video save_path is {save_path}")
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )

    df = pd.DataFrame()
    df_person = pd.DataFrame()

    while True:
        ret_val, frame = cap.read()
        if ret_val:
            frame_count += 1
            if frame_count % save_interval == 0:
                timestamp = frame_count / fps
                new_frame = preprocess_frame(frame)
                outputs, img_info = predictor.inference(new_frame)

                result_frame, temp_df, temp_df_person = predictor.visual(outputs[0], img_info, predictor.confthre, timestamp)
                
                if temp_df is not None:
                    df = temp_df
                if temp_df_person is not None:
                    df_person = temp_df_person

                if args.save_result:
                    vid_writer.write(result_frame)
                else:
                    cv2.namedWindow("yolox", cv2.WINDOW_NORMAL)
                    cv2.imshow("yolox", result_frame)
                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    print(df)
                    print(df_person)
                    break
        else:
            video_filename = os.path.basename(save_path)
            csv_filename = os.path.splitext(video_filename)[0] + ".xlsx"
            csv_person_filename = os.path.splitext(video_filename)[0] + "_person.xlsx"

            df[['tempo', 'corpo_xy', 'width', 'height', 'area',
                'status', 'proportion', 'cabeca_xy']].to_excel(os.path.join(save_folder, csv_filename),index=False)
            
            df_person[['tempo', 'status']].to_excel(os.path.join(save_folder, csv_person_filename), index=False)

            classificar_movimento(save_folder, csv_filename, csv_person_filename, 8, 0.1)

            cap.release()
            vid_writer.release()
            
            # Deletar o resultado da inferencia
            os.remove(os.path.join(save_folder, video_filename))

            # Deletar o video utilizado
            # os.remove(args.path)
            destination_path = os.path.join(save_folder, video_filename)
            shutil.copy(args.path, destination_path)
            
            break


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(
        model, exp, COCO_CLASSES, trt_file, decoder,
        args.device, args.fp16, args.legacy,
    )
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, args.path, current_time, args.save_result)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
