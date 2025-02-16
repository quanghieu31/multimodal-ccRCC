from tqdm import tqdm
import openslide




def tile_slide(
    in_svs: str,
    target_size: int = 256,
    max_intensity: float = 0.85,
    min_intensity: float = 0.15,
    target_magnification: int = 20,
    use_tqdm: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:

    """
    Tile a slide into tiles of size target_size x target_size. 
    Tiles are only kept if the mean intensity of the tile is between min_intensity and max_intensity.

    "tile a whole slide, may need to just add some logic to write the tiles to disk as soon as we read them, 
    so that we don't hold the whole image in memory"
    """

    slide = openslide.open_slide(in_svs)
    if "openslide.objective-power" not in slide.properties:
        return dummy_return(target_size)
    magnification = int(slide.properties["openslide.objective-power"])
    scale = magnification / target_magnification
    size = int(target_size * scale)
    resize = lambda x: x
    if size != target_size:
        resize = transforms.Resize(target_size)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    width, height = slide.level_dimensions[0]
    coords = []
    for x in range(0, width, size):
        for y in range(0, height, size):
            if x + size > width or y + size > height:
                continue
            coords.append((x, y))

    tiles = []
    used_coords = []
    for x, y in tqdm(coords, desc="Tiles", disable=not use_tqdm):
        try:
            tile = slide.read_region((x, y), 0, (size, size)).convert("RGB")
        except openslide.OpenSlideError as e:
            print(f"Skipping {in_svs}, caught following error")
            print(e)
            return dummy_return(target_size)
        tile = resize(tile)
        gray = tile.convert("L")
        intensity = (np.asarray(gray) / 255).mean()
        if intensity > max_intensity or intensity < min_intensity:
            continue
        tile = transform(tile)
        tiles.append(tile)
        used_coords.append((x, y))
    return torch.stack(tiles), torch.as_tensor(used_coords)