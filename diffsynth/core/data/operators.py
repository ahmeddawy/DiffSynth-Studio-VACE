import torch, torchvision, imageio, os
import numpy as np
import imageio.v3 as iio
from PIL import Image


class DataProcessingPipeline:
    def __init__(self, operators=None):
        self.operators: list[DataProcessingOperator] = [] if operators is None else operators
        
    def __call__(self, data):
        for operator in self.operators:
            data = operator(data)
        return data
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline(self.operators + pipe.operators)


class DataProcessingOperator:
    def __call__(self, data):
        raise NotImplementedError("DataProcessingOperator cannot be called directly.")
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline([self]).__rshift__(pipe)


class DataProcessingOperatorRaw(DataProcessingOperator):
    def __call__(self, data):
        return data


class ToInt(DataProcessingOperator):
    def __call__(self, data):
        return int(data)


class ToFloat(DataProcessingOperator):
    def __call__(self, data):
        return float(data)


class ToStr(DataProcessingOperator):
    def __init__(self, none_value=""):
        self.none_value = none_value
    
    def __call__(self, data):
        if data is None: data = self.none_value
        return str(data)


class LoadImage(DataProcessingOperator):
    def __init__(self, convert_RGB=True, convert_RGBA=False):
        self.convert_RGB = convert_RGB
        self.convert_RGBA = convert_RGBA
    
    def __call__(self, data: str):
        image = Image.open(data)
        if self.convert_RGB: image = image.convert("RGB")
        if self.convert_RGBA: image = image.convert("RGBA")
        return image


class ImageCropAndResize(DataProcessingOperator):
    def __init__(self, height=None, width=None, max_pixels=None, height_division_factor=1, width_division_factor=1):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    def get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    def __call__(self, data: Image.Image):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image


class ToList(DataProcessingOperator):
    def __call__(self, data):
        return [data]
    

class LoadVideo(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str):
        reader = imageio.get_reader(data)
        num_frames = self.get_num_frames(reader)
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.frame_processor(frame)
            frames.append(frame)
        reader.close()
        return frames


class SequencialProcess(DataProcessingOperator):
    def __init__(self, operator=lambda x: x):
        self.operator = operator
        
    def __call__(self, data):
        return [self.operator(i) for i in data]


class LoadGIF(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        
    def get_num_frames(self, path):
        num_frames = self.num_frames
        images = iio.imread(path, mode="RGB")
        if len(images) < num_frames:
            num_frames = len(images)
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str):
        num_frames = self.get_num_frames(data)
        frames = []
        images = iio.imread(data, mode="RGB")
        for img in images:
            frame = Image.fromarray(img)
            frame = self.frame_processor(frame)
            frames.append(frame)
            if len(frames) >= num_frames:
                break
        return frames


class RouteByExtensionName(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data: str):
        file_ext_name = data.split(".")[-1].lower()
        for ext_names, operator in self.operator_map:
            if ext_names is None or file_ext_name in ext_names:
                return operator(data)
        raise ValueError(f"Unsupported file: {data}")


class RouteByType(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data):
        for dtype, operator in self.operator_map:
            if dtype is None or isinstance(data, dtype):
                return operator(data)
        raise ValueError(f"Unsupported data: {data}")


class LoadTorchPickle(DataProcessingOperator):
    def __init__(self, map_location="cpu"):
        self.map_location = map_location
        
    def __call__(self, data):
        return torch.load(data, map_location=self.map_location, weights_only=False)


class ToAbsolutePath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path
        
    def __call__(self, data):
        return os.path.join(self.base_path, data)


class LoadAudio(DataProcessingOperator):
    def __init__(self, sr=16000):
        self.sr = sr
    def __call__(self, data: str):
        import librosa
        input_audio, sample_rate = librosa.load(data, sr=self.sr)
        return input_audio


class VideoAugment:
    def __init__(
        self,
        geo_keys=None,
        mask_keys=None,
        color_keys=None,
        hflip_prob=0.0,
        color_jitter_prob=0.0,
        color_jitter_strength=0.2,
        fog_prob=0.0,
        rain_prob=0.0,
        snow_prob=0.0,
        sunflare_prob=0.0,
    ):
        try:
            import albumentations as A
        except Exception as exc:
            raise ImportError("Albumentations is required for VideoAugment. Please install it to use augmentation.") from exc

        self.A = A
        self.geo_keys = tuple(geo_keys or [])
        self.mask_keys = set(mask_keys or [])
        self.color_keys = tuple(color_keys or [])
        self.hflip_prob = float(hflip_prob)
        self.color_jitter_prob = float(color_jitter_prob)
        self.color_jitter_strength = float(color_jitter_strength)
        self.fog_prob = float(fog_prob)
        self.rain_prob = float(rain_prob)
        self.snow_prob = float(snow_prob)
        self.sunflare_prob = float(sunflare_prob)
        self.geo_aug = self._build_geo_aug()
        self.color_aug = self._build_color_aug()
        self.enabled = self.geo_aug is not None or self.color_aug is not None

    def _build_geo_aug(self):
        transforms = []
        if self.hflip_prob > 0:
            transforms.append(self.A.HorizontalFlip(p=self.hflip_prob))
        if not transforms:
            return None
        return self.A.ReplayCompose(transforms)

    def _build_color_aug(self):
        transforms = []
        if self.color_jitter_prob > 0:
            strength = self.color_jitter_strength
            transforms.append(
                self.A.ColorJitter(
                    brightness=strength,
                    contrast=strength,
                    saturation=strength,
                    hue=min(0.1, strength),
                    p=self.color_jitter_prob,
                )
            )
        if self.fog_prob > 0:
            if not hasattr(self.A, "RandomFog"):
                raise ImportError("Albumentations RandomFog is unavailable; upgrade albumentations to use fog augmentation.")
            transforms.append(self.A.RandomFog(p=self.fog_prob))
        if self.rain_prob > 0:
            if not hasattr(self.A, "RandomRain"):
                raise ImportError("Albumentations RandomRain is unavailable; upgrade albumentations to use rain augmentation.")
            transforms.append(self.A.RandomRain(p=self.rain_prob))
        if self.snow_prob > 0:
            if not hasattr(self.A, "RandomSnow"):
                raise ImportError("Albumentations RandomSnow is unavailable; upgrade albumentations to use snow augmentation.")
            transforms.append(self.A.RandomSnow(p=self.snow_prob))
        if self.sunflare_prob > 0:
            if not hasattr(self.A, "RandomSunFlare"):
                raise ImportError("Albumentations RandomSunFlare is unavailable; upgrade albumentations to use sunflare augmentation.")
            transforms.append(self.A.RandomSunFlare(p=self.sunflare_prob))
        if not transforms:
            return None
        return self.A.ReplayCompose(transforms)

    def _ensure_list(self, value):
        if value is None:
            return None
        if isinstance(value, list):
            return value
        return [value]

    def _pick_reference_frame(self, sample, keys):
        for key in keys:
            frames = self._ensure_list(sample.get(key))
            if not frames:
                continue
            return frames[0]
        return None

    def _apply_geo_to_frames(self, frames, replay, is_mask=False):
        out = []
        for frame in frames:
            arr = np.array(frame)
            if is_mask:
                if arr.ndim == 3:
                    mask = arr[:, :, 0]
                    image = arr
                else:
                    mask = arr
                    image = np.repeat(arr[:, :, None], 3, axis=2)
                result = self.A.ReplayCompose.replay(replay, image=image, mask=mask)
                mask_out = result["mask"]
                if mask_out.ndim == 2:
                    mask_out = np.repeat(mask_out[:, :, None], 3, axis=2)
                mask_out = mask_out.astype(np.uint8)
                out.append(Image.fromarray(mask_out))
            else:
                if arr.ndim == 2:
                    arr = np.repeat(arr[:, :, None], 3, axis=2)
                result = self.A.ReplayCompose.replay(replay, image=arr)
                image_out = result["image"].astype(np.uint8)
                out.append(Image.fromarray(image_out))
        return out

    def _apply_color_to_frames(self, frames, replay):
        out = []
        for frame in frames:
            arr = np.array(frame)
            if arr.ndim == 2:
                arr = np.repeat(arr[:, :, None], 3, axis=2)
            result = self.A.ReplayCompose.replay(replay, image=arr)
            image_out = result["image"].astype(np.uint8)
            out.append(Image.fromarray(image_out))
        return out

    def __call__(self, sample: dict):
        if not self.enabled:
            return sample
        data = sample.copy()
        if self.geo_aug is not None:
            geo_keys = list(self.geo_keys)
            for key in self.mask_keys:
                if key not in geo_keys:
                    geo_keys.append(key)
            ref = self._pick_reference_frame(data, geo_keys)
            if ref is not None:
                geo_replay = self.geo_aug(image=np.array(ref))["replay"]
                for key in geo_keys:
                    frames = self._ensure_list(data.get(key))
                    if not frames:
                        continue
                    data[key] = self._apply_geo_to_frames(frames, geo_replay, is_mask=key in self.mask_keys)
        if self.color_aug is not None:
            ref = self._pick_reference_frame(data, self.color_keys)
            if ref is not None:
                color_replay = self.color_aug(image=np.array(ref))["replay"]
                for key in self.color_keys:
                    frames = self._ensure_list(data.get(key))
                    if not frames:
                        continue
                    data[key] = self._apply_color_to_frames(frames, color_replay)
        return data
