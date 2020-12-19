import numpy as np
import torch
from PIL import Image
import cv2


class DetectAugment(object):
    def __init__(self):
        super(DetectAugment, self).__init__()

    def get_transform(self, img):
        pass

    def __call__(self, img: np.ndarray, labels: np.ndarray):
        """
        :param img: np.ndarray (RGB) 模型 0-255
        :param labels: [box_num,6] (weights,label_idx,x1,y1,x2,y2) 其中坐标为原图上坐标(不需要normalize到[0，1])
        :return:
        """
        transform = self.get_transform(img)
        img=transform.apply_image(img)
        labels[:,2:]=transform.apply_box(labels[:,2:])
        return img, labels




###################################################################################################
class AffineTransform(object):
    """
    Augmentation from CenterNet
    """

    def __init__(self, src, dst, output_size):
        """
        output_size:(w, h)
        """
        super().__init__()
        self.output_size=output_size
        self.affine = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Apply AffineTransform for the image(s).

        Args:
            img (ndarray): of shape HxW, HxWxC, or NxHxWxC. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: the image(s) after applying affine transform.
        """
        return cv2.warpAffine(img, self.affine, self.output_size, flags=cv2.INTER_LINEAR)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Affine the coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            ndarray: the flipped coordinates.

        Note:
            The inputs are floating point coordinates, not pixel indices.
            Therefore they are flipped by `(W - x, H - y)`, not
            `(W - 1 - x, H 1 - y)`.
        """
        # aug_coord (N, 3) shape, self.affine (2, 3) shape
        w, h = self.output_size
        aug_coords = np.column_stack((coords, np.ones(coords.shape[0])))
        coords = np.dot(aug_coords, self.affine.T)
        coords[..., 0] = np.clip(coords[..., 0], 0, w - 1)
        coords[..., 1] = np.clip(coords[..., 1], 0, h - 1)
        return coords

    def apply_box(self, box: np.ndarray) -> np.ndarray:
        """
        Apply the transform on an axis-aligned box.
        By default will transform the corner points and use their
        minimum/maximum to create a new axis-aligned box.
        Note that this default may change the size of your box, e.g. in
        rotations.

        Args:
            box (ndarray): Nx4 floating point array of XYXY format in absolute coordinates.
        Returns:
            ndarray: box after apply the transformation.

        Note:
            The coordinates are not pixel indices. Coordinates on an image of
            shape (H, W) are in range [0, W] or [0, H].
        """
        # Indexes of converting (x0, y0, x1, y1) box into 4 coordinates of
        # ([x0, y0], [x1, y0], [x0, y1], [x1, y1]).
        idxs = np.array([(0, 1), (2, 1), (0, 3), (2, 3)]).flatten()
        coords = np.asarray(box).reshape(-1, 4)[:, idxs].reshape(-1, 2)
        coords = self.apply_coords(coords).reshape((-1, 4, 2))
        minxy = coords.min(axis=1)
        maxxy = coords.max(axis=1)
        trans_boxes = np.concatenate((minxy, maxxy), axis=1)

        return trans_boxes




class CenterAffine(DetectAugment):
    """
    Affine Transform for CenterNet
    """

    def __init__(self, boarder, output_size, random_aug=True):
        """
        Args:
            boarder(int): boarder size of image
            output_size(tuple): a tuple represents (width, height) of image
            random_aug(bool): whether apply random augmentation on annos or not
        """
        super().__init__()
        self.boarder = boarder
        self.output_size = output_size
        self.random_aug = random_aug

    def get_transform(self, img):
        """
        generate one `AffineTransform` for input image
        """
        img_shape = img.shape[:2]
        center, scale = self.generate_center_and_scale(img_shape)
        src, dst = self.generate_src_and_dst(center, scale, self.output_size)
        return AffineTransform(src, dst, self.output_size)

    @staticmethod
    def _get_boarder(boarder, size):
        """
        decide the boarder size of image
        """
        # NOTE This func may be reimplemented in the future
        i = 1
        size //= 2
        while size <= boarder // i:
            i *= 2
        return boarder // i

    def generate_center_and_scale(self, img_shape):
        r"""
        generate center and scale for image randomly

        Args:
            shape(tuple): a tuple represents (height, width) of image
        """
        height, width = img_shape
        center = np.array([width / 2, height / 2], dtype=np.float32)
        scale = float(max(img_shape))
        if self.random_aug:
            self.scale_factor = np.random.choice(np.arange(0.6, 1.4, 0.1))
            scale = scale * self.scale_factor
            h_boarder = self._get_boarder(self.boarder, height)
            w_boarder = self._get_boarder(self.boarder, width)
            center[0] = np.random.randint(low=w_boarder, high=width - w_boarder)
            center[1] = np.random.randint(low=h_boarder, high=height - h_boarder)
        else:
            raise NotImplementedError("Non-random augmentation not implemented")

        return center, scale

    @staticmethod
    def generate_src_and_dst(center, size, output_size):
        r"""
        generate source and destination for affine transform
        """
        if not isinstance(size, np.ndarray) and not isinstance(size, list):
            size = np.array([size, size], dtype=np.float32)
        src = np.zeros((3, 2), dtype=np.float32)
        src_w = size[0]
        src_dir = [0, src_w * -0.5]
        src[0, :] = center
        src[1, :] = src[0, :] + src_dir
        src[2, :] = src[1, :] + (src_dir[1], -src_dir[0])

        dst = np.zeros((3, 2), dtype=np.float32)
        dst_w, dst_h = output_size
        dst_dir = [0, dst_w * -0.5]
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = dst[0, :] + dst_dir
        dst[2, :] = dst[1, :] + (dst_dir[1], -dst_dir[0])

        return src, dst


    def __call__(self, img: np.ndarray, labels: np.ndarray):
        """
        :param img: np.ndarray (RGB) 模型 0-255
        :param labels: [box_num,6] (weights,label_idx,x1,y1,x2,y2) 其中坐标为原图上坐标(不需要normalize到[0，1])
        :return:
        """
        transform = self.get_transform(img)
        img=transform.apply_image(img)
        temp_label=transform.apply_box(labels[:,2:])  #[x1,y1,x2,y2]
        w = temp_label[:,2]-temp_label[:,0]
        h = temp_label[:,3]-temp_label[:,1]
        area = w*h
        area0 = (labels[:, 4] - labels[:, 2]) * (labels[:, 5] - labels[:, 3])
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
        i = (w > 2) & (h > 2) & (area / (area0 * self.scale_factor + 1e-16) > 0.2) & (ar < 20)
        labels = labels[i]
        labels[:, 2:6] = temp_label[i]

        return img, labels




##############################################################################################

class BlendTransform(object):
    """
    Transforms pixel colors with PIL enhance functions.
    """

    def __init__(
        self, src_image: np.ndarray, src_weight: float, dst_weight: float
    ):
        """
        Blends the input image (dst_image) with the src_image using formula:
        ``src_weight * src_image + dst_weight * dst_image``

        Args:
            src_image (ndarray): Input image is blended with this image
            src_weight (float): Blend weighting of src_image
            dst_weight (float): Blend weighting of dst_image
        """
        super().__init__()
        self.src_image = src_image
        self.src_weight = src_weight
        self.dst_weight = dst_weight

    def apply_image(self, img: np.ndarray, interp: str = None) -> np.ndarray:
        """
        Apply blend transform on the image(s).

        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
            interp (str): keep this option for consistency, perform blend would not
                require interpolation.
        Returns:
            ndarray: blended image(s).
        """
        if img.dtype == np.uint8:
            img = img.astype(np.float32)
            img = self.src_weight * self.src_image + self.dst_weight * img
            return np.clip(img, 0, 255).astype(np.uint8)
        else:
            return self.src_weight * self.src_image + self.dst_weight * img

    def apply_box(self, box: np.ndarray) -> np.ndarray:
        return box





class RandomContrast(DetectAugment):
    """
    Randomly transforms image contrast.

    Contrast intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce contrast
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase contrast

    See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
    """

    def __init__(self, intensity_min, intensity_max):
        """
        Args:
            intensity_min (float): Minimum augmentation
            intensity_max (float): Maximum augmentation
        """
        super().__init__()
        self.intensity_min=intensity_min
        self.intensity_max=intensity_max

    def get_transform(self, img):
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        return BlendTransform(src_image=img.mean(), src_weight=1 - w, dst_weight=w)



class RandomBrightness(DetectAugment):
    """
    Randomly transforms image brightness.

    Brightness intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce brightness
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase brightness

    See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
    """

    def __init__(self, intensity_min, intensity_max):
        """
        Args:
            intensity_min (float): Minimum augmentation
            intensity_max (float): Maximum augmentation
        """
        super().__init__()
        self.intensity_min = intensity_min
        self.intensity_max = intensity_max

    def get_transform(self, img):
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        return BlendTransform(src_image=0, src_weight=1 - w, dst_weight=w)


class RandomSaturation(DetectAugment):
    """
    Randomly transforms image saturation.

    Saturation intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce saturation (make the image more grayscale)
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase saturation

    See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
    """

    def __init__(self, intensity_min, intensity_max):
        """
        Args:
            intensity_min (float): Minimum augmentation (1 preserves input).
            intensity_max (float): Maximum augmentation (1 preserves input).
        """
        super().__init__()
        self.intensity_min = intensity_min
        self.intensity_max = intensity_max

    def get_transform(self, img):
        assert img.shape[-1] == 3, "Saturation only works on RGB images"
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        grayscale = img.dot([0.299, 0.587, 0.114])[:, :, np.newaxis]
        return BlendTransform(src_image=grayscale, src_weight=1 - w, dst_weight=w)


class RandomLighting(DetectAugment):
    """
    Randomly transforms image color using fixed PCA over ImageNet.

    The degree of color jittering is randomly sampled via a normal distribution,
    with standard deviation given by the scale parameter.
    """

    def __init__(self, scale):
        """
        Args:
            scale (float): Standard deviation of principal component weighting.
        """
        super().__init__()
        self.scale=scale
        self.eigen_vecs = np.array(
            [
                [-0.5675, 0.7192, 0.4009],
                [-0.5808, -0.0045, -0.8140],
                [-0.5836, -0.6948, 0.4203],
            ]
        )
        self.eigen_vals = np.array([0.2175, 0.0188, 0.0045])

    def get_transform(self, img):
        assert img.shape[-1] == 3, "Saturation only works on RGB images"
        weights = np.random.normal(scale=self.scale, size=3)
        return BlendTransform(
            src_image=self.eigen_vecs.dot(weights * self.eigen_vals),
            src_weight=1.0,
            dst_weight=1.0,
        )



#######################################################################################



class HFlipTransform(object):
    """
    Perform horizontal flip.
    """

    def __init__(self, width: int):
        super().__init__()
        self.width = width

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Flip the image(s).

        Args:
            img (ndarray): of shape HxW, HxWxC, or NxHxWxC. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: the flipped image(s).
        """
        tensor = torch.from_numpy(np.ascontiguousarray(img))
        if len(tensor.shape) == 2:
            # For dimension of HxW.
            tensor = tensor.flip((-1))
        elif len(tensor.shape) > 2:
            # For dimension of HxWxC, NxHxWxC.
            tensor = tensor.flip((-2))
        return tensor.numpy()

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Flip the coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            ndarray: the flipped coordinates.

        Note:
            The inputs are floating point coordinates, not pixel indices.
            Therefore they are flipped by `(W - x, H - y)`, not
            `(W - 1 - x, H 1 - y)`.
        """
        coords[:, 0] = self.width - coords[:, 0]
        return coords

    def apply_box(self, box: np.ndarray) -> np.ndarray:
        """
        Apply the transform on an axis-aligned box.
        By default will transform the corner points and use their
        minimum/maximum to create a new axis-aligned box.
        Note that this default may change the size of your box, e.g. in
        rotations.

        Args:
            box (ndarray): Nx4 floating point array of XYXY format in absolute
                coordinates.
        Returns:
            ndarray: box after apply the transformation.

        Note:
            The coordinates are not pixel indices. Coordinates on an image of
            shape (H, W) are in range [0, W] or [0, H].
        """
        # Indexes of converting (x0, y0, x1, y1) box into 4 coordinates of
        # ([x0, y0], [x1, y0], [x0, y1], [x1, y1]).
        idxs = np.array([(0, 1), (2, 1), (0, 3), (2, 3)]).flatten()
        coords = np.asarray(box).reshape(-1, 4)[:, idxs].reshape(-1, 2)
        coords = self.apply_coords(coords).reshape((-1, 4, 2))
        minxy = coords.min(axis=1)
        maxxy = coords.max(axis=1)
        trans_boxes = np.concatenate((minxy, maxxy), axis=1)
        return trans_boxes


class NoOpTransform(object):
    """
    A transform that does nothing.
    """

    def __init__(self):
        super().__init__()

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        return img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_box(self, box: np.ndarray) -> np.ndarray:
        return box


class RandomFlip(DetectAugment):
    """
    Flip the image horizontally with the given probability.

    TODO Vertical flip to be implemented.
    """

    def __init__(self, prob=0.5):
        """
        Args:
            prob (float): probability of flip.
        """
        # TODO implement vertical flip when we need it
        super().__init__()
        self.prob = prob


    def get_transform(self, img):
        _, w = img.shape[:2]
        do = np.random.uniform(0.,1.) < self.prob
        if do:
            return HFlipTransform(w)
        else:
            return NoOpTransform()





###################################################################################################
class ResizeTransform(object):
    """
    Resize the image to a target size.
    """

    def __init__(self, h, w, new_h, new_w, interp):
        """
        Args:
            h, w (int): original image size
            new_h, new_w (int): new image size
            interp: PIL interpolation methods
        """
        # TODO decide on PIL vs opencv
        super().__init__()
        self.h=h
        self.w=w
        self.new_h=new_h
        self.new_w=new_w
        self.interp=interp

    def apply_image(self, img, interp=None):
        assert img.shape[:2] == (self.h, self.w)
        pil_image = Image.fromarray(img)
        interp_method = interp if interp is not None else self.interp
        pil_image = pil_image.resize((self.new_w, self.new_h), interp_method)
        ret = np.asarray(pil_image)
        return ret

    def apply_coords(self, coords):
        coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords

    def apply_box(self, box: np.ndarray) -> np.ndarray:
        """
        Apply the transform on an axis-aligned box.
        By default will transform the corner points and use their
        minimum/maximum to create a new axis-aligned box.
        Note that this default may change the size of your box, e.g. in
        rotations.

        Args:
            box (ndarray): Nx4 floating point array of XYXY format in absolute
                coordinates.
        Returns:
            ndarray: box after apply the transformation.

        Note:
            The coordinates are not pixel indices. Coordinates on an image of
            shape (H, W) are in range [0, W] or [0, H].
        """
        # Indexes of converting (x0, y0, x1, y1) box into 4 coordinates of
        # ([x0, y0], [x1, y0], [x0, y1], [x1, y1]).
        idxs = np.array([(0, 1), (2, 1), (0, 3), (2, 3)]).flatten()
        coords = np.asarray(box).reshape(-1, 4)[:, idxs].reshape(-1, 2)
        coords = self.apply_coords(coords).reshape((-1, 4, 2))
        minxy = coords.min(axis=1)
        maxxy = coords.max(axis=1)
        trans_boxes = np.concatenate((minxy, maxxy), axis=1)
        return trans_boxes




class ResizeShortestEdge(DetectAugment):
    """
    Scale the shorter edge to the given size, with a limit of `max_size` on the longer edge.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
    """

    def __init__(
        self,
        short_edge_length,
        max_size,
        sample_style="range",
        interp=Image.BILINEAR,
    ):
        """
        Args:
            short_edge_length (list[int]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the shortest edge length.
                If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
            max_size (int): maximum allowed longest edge length.
            sample_style (str): either "range" or "choice".
        """
        super().__init__()
        assert sample_style in ["range", "choice"], sample_style

        self.is_range = sample_style == "range"
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        self.short_edge_length=short_edge_length
        self.max_size=max_size
        self.sample_style=sample_style
        self.interp=interp

    def get_transform(self, img):
        h, w = img.shape[:2]

        if self.is_range:
            size = np.random.randint(
                self.short_edge_length[0], self.short_edge_length[1] + 1
            )
        else:
            size = np.random.choice(self.short_edge_length)
        if size == 0:
            return NoOpTransform()

        scale = size * 1.0 / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return ResizeTransform(h, w, newh, neww, self.interp)




###################################################################################################

class ScalePadding(object):
    """
    等比缩放长边至指定尺寸，padding短边部分
    """

    def __init__(self, target_size=(640, 640),
                 padding_val=(114, 114, 114),
                 minimum_rectangle=False,
                 scale_up=True, **kwargs):
        super(ScalePadding, self).__init__(**kwargs)
        self.p = 1
        self.new_shape = target_size
        self.padding_val = padding_val
        self.minimum_rectangle = minimum_rectangle
        self.scale_up = scale_up

    def make_border(self, img: np.ndarray):
        # h,w
        shape = img.shape[:2]
        if isinstance(self.new_shape, int):
            self.new_shape = (self.new_shape, self.new_shape)
        r = min(self.new_shape[1] / shape[0], self.new_shape[0] / shape[1])
        if not self.scale_up:
            r = min(r, 1.0)
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = self.new_shape[0] - new_unpad[0], self.new_shape[1] - new_unpad[1]
        if self.minimum_rectangle:
            dw, dh = np.mod(dw, 64), np.mod(dh, 64)

        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.padding_val)
        return img, ratio, (left, top)

    def img_aug(self, img: np.ndarray) -> np.ndarray:
        img, _, _ = self.make_border(img)
        return img

    def aug(self, img: np.ndarray, labels: np.ndarray) -> tuple:
        img, ratio, (left, top) = self.make_border(img)
        if labels.shape[0] > 0:
            labels[:, [2, 4]] = ratio[0] * labels[:, [2, 4]] + left
            labels[:, [3, 5]] = ratio[1] * labels[:, [3, 5]] + top
        return img, labels

    def __call__(self, img: np.ndarray, labels: np.ndarray):
        """
        :param img: np.ndarray (RGB) 模型 0-255
        :param labels: [box_num,6] (weights,label_idx,x1,y1,x2,y2) 其中坐标为原图上坐标(不需要normalize到[0，1])
        :return:
        """
        img,labels=self.aug(img,labels)
        return img, labels




###############################################################################################


class Compose(object):
    """
    串行数据增强的方式
    """

    def __init__(self, transforms):
        super(Compose, self).__init__()
        self.transforms = transforms

    def __call__(self, img: np.ndarray, labels: np.ndarray):
        """
        :param img: np.ndarray (RGB) 模型 0-255
        :param labels: [box_num,6] (weights,label_idx,x1,y1,x2,y2) 其中坐标为原图上坐标(不需要normalize到[0，1])
        :return:
        """
        for transform in self.transforms:
            img, labels = transform(img, labels)
        return img, labels



