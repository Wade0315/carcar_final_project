from pathlib import Path
import argparse
import re
import uuid


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
FINAL_NAME_PATTERN = re.compile(r"^image_(\d+)$")


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def get_final_index(path: Path) -> int | None:
    match = FINAL_NAME_PATTERN.fullmatch(path.stem)
    if not match:
        return None
    return int(match.group(1))


def collect_rename_plan(stock_dir: Path) -> list[tuple[Path, Path]]:
    images = sorted(
        (path for path in stock_dir.iterdir() if is_image(path)),
        key=lambda path: (path.stat().st_mtime, path.name.lower()),
    )

    used_indices = {
        index
        for image in images
        if (index := get_final_index(image)) is not None
    }

    next_index = max(used_indices, default=-1) + 1
    rename_plan = []

    for image in images:
        if get_final_index(image) is not None:
            continue

        while next_index in used_indices:
            next_index += 1

        target = image.with_name(f"image_{next_index}{image.suffix.lower()}")
        used_indices.add(next_index)
        rename_plan.append((image, target))
        next_index += 1

    return rename_plan


def collect_reorder_plan(stock_dir: Path) -> list[tuple[Path, Path]]:
    images = sorted(
        (path for path in stock_dir.iterdir() if is_image(path)),
        key=lambda path: (
            get_final_index(path) is None,
            get_final_index(path) if get_final_index(path) is not None else path.stat().st_mtime,
            path.name.lower(),
        ),
    )

    reorder_plan = []

    for index, image in enumerate(images):
        target = image.with_name(f"image_{index}{image.suffix.lower()}")
        if image.name != target.name:
            reorder_plan.append((image, target))

    return reorder_plan


def apply_rename_plan(rename_plan: list[tuple[Path, Path]]) -> None:
    temp_plan = []

    for source, target in rename_plan:
        temp = source.with_name(f".rename_tmp_{uuid.uuid4().hex}{source.suffix}")
        source.rename(temp)
        temp_plan.append((temp, target))

    for temp, target in temp_plan:
        temp.rename(target)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rename photos in stock to image_i without reusing existing indices."
    )
    parser.add_argument(
        "--stock-dir",
        type=Path,
        default=Path(__file__).parent / "stock",
        help="Folder containing photos. Default: photo_test/stock",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be renamed without changing files.",
    )
    parser.add_argument(
        "--reorder",
        action="store_true",
        help="Reorder all photos to continuous image_i names starting from 0.",
    )
    args = parser.parse_args()

    stock_dir = args.stock_dir.resolve()
    if not stock_dir.exists():
        raise FileNotFoundError(f"Stock folder not found: {stock_dir}")
    if not stock_dir.is_dir():
        raise NotADirectoryError(f"Stock path is not a folder: {stock_dir}")

    rename_plan = (
        collect_reorder_plan(stock_dir) if args.reorder else collect_rename_plan(stock_dir)
    )
    if not rename_plan:
        print("No photos need renaming.")
        return

    for source, target in rename_plan:
        print(f"{source.name} -> {target.name}")

    if args.dry_run:
        print(f"Dry run only. {len(rename_plan)} photo(s) would be renamed.")
        return

    apply_rename_plan(rename_plan)
    print(f"Renamed {len(rename_plan)} photo(s).")


if __name__ == "__main__":
    main()
