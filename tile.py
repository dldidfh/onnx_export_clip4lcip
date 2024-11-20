from typing import Literal
from dataclasses import dataclass

import numpy as np

tile_dict = {
    "no_tile": {"px": 10000, "ratio": 0.25},
    "L": {"px": 672, "ratio": 0.25},
    "M": {"px": 560, "ratio": 0.25},
    "S": {"px": 448, "ratio": 0.25},
}


@dataclass
class Tile:
    """
    Tile attribute. size attribute must be "L" or "M" or "S".

    Example:
        tile_s = Tile("S")
        tile_s
            >>> Tile(size='S', px=448, overlap=0.25)
        tile_s.size
            >>> 'S'
        tile_s.px
            >>> 448
    """

    size: str
    px: int
    overlap: float

    def __init__(self, size: str):
        if size is None:
            size = "no_tile"
        elif isinstance(size, Tile):
            size = size.size
        if size in tile_dict:
            self.size = size
            self.px = tile_dict[size]["px"]
            self.overlap = tile_dict[size]["ratio"]
        else:
            raise ValueError(f"Size '{self.size}' not found in tile_dict.")


def tile_videos(
    videos_sequenced: np.ndarray, tile_size: Literal["L", "M", "S"] = "S"
) -> np.ndarray:
    if len(videos_sequenced.shape) != 5:
        raise ValueError(
            f"Video shape must be (Video batch size, Sequence length, Height, Width, Channel). Your current video shape is {videos_sequenced.shape}."
        )

    videos_tiled_sequenced = []
    for video in videos_sequenced:
        sequence_tiled = tile_sequence(
            sequence=video,
            tile_size=tile_size,
        )
        tiles_sequenced = np.swapaxes(sequence_tiled, 0, 1)
        videos_tiled_sequenced.append(tiles_sequenced)

    videos_tiled_sequenced = np.array(videos_tiled_sequenced)

    return videos_tiled_sequenced


def tile_sequence(
    sequence: np.ndarray, tile_size: Literal["L", "M", "S"] = "S"
) -> np.ndarray:
    if len(sequence.shape) != 4:
        raise ValueError(
            f"Image shape must be (Image batch size, Height, Width, Channel). Your current image shape is {sequence.shape}."
        )

    sequence_tiled = []
    for frame in sequence:
        tiles = tile_frame(frame, tile_size)
        sequence_tiled.append(tiles)

    sequence_tiled = np.array(sequence_tiled)
    return sequence_tiled


def tile_frame(
    frame: np.ndarray, tile_size: Literal["L", "M", "S"] = "S"
) -> np.ndarray:
    if len(frame.shape) != 3:
        raise ValueError(
            f"Image shape must be (Height, Width, Channel). Your current image shape is {frame.shape}."
        )
    tiles = []
    tile = Tile(tile_size)
    height, width, channel = frame.shape
    # if tile.px > height:
    #     raise ValueError(f"The tile size must be less than the height of the image.")
    # if tile.px > width:
    #     raise ValueError(f"The tile size must be less than the width of the image.")

    # tiles.append(cv2.resize(frame, (tile.px, tile.px),interpolation=cv2.INTER_LINEAR))

    if height < tile.px or width < tile.px:
        tiles.append(frame)
        return np.array(tiles)

    overlap_size = int(tile.px * tile.overlap)
    tile_stride = int(tile.px - overlap_size)

    num_tiles_vertical = (height - tile.px) // tile_stride + 1
    num_tiles_horizontal = (width - tile.px) // tile_stride + 1

    for i in range(num_tiles_vertical):
        for j in range(num_tiles_horizontal):
            start_y = i * tile_stride
            start_x = j * tile_stride
            _tile = frame[start_y : start_y + tile.px, start_x : start_x + tile.px, :]
            print(f"원본 : x : {start_x}, y : {start_y}")
            tiles.append(_tile)

    has_abandoned_vertical = True if (height - tile.px) % tile_stride > 0 else False
    has_abandoned_horizontal = True if (width - tile.px) % tile_stride > 0 else False

    if has_abandoned_vertical:
        for j in range(num_tiles_horizontal):
            start_y = height - tile.px
            start_x = j * tile_stride
            _tile = frame[start_y : start_y + tile.px, start_x : start_x + tile.px, :]
            print(f"hirizontal : x : {start_x}, y : {start_y}")
            tiles.append(_tile)

    if has_abandoned_horizontal:
        for i in range(num_tiles_vertical):
            start_y = i * tile_stride
            start_x = width - tile.px
            _tile = frame[start_y : start_y + tile.px, start_x : start_x + tile.px, :]
            print(f"vertical : x : {start_x}, y : {start_y}")
            tiles.append(_tile)

    if has_abandoned_vertical and has_abandoned_horizontal:
        start_y = height - tile.px
        start_x = width - tile.px
        _tile = frame[start_y : start_y + tile.px, start_x : start_x + tile.px, :]
        print(f"last 모서리 : x : {start_x}, y : {start_y}")
        tiles.append(_tile)

    return np.array(tiles)