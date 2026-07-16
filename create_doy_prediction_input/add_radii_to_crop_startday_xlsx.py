import argparse
import csv
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET


SPREADSHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS = {"ss": SPREADSHEET_NS}
ET.register_namespace("", SPREADSHEET_NS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add radius_x and radius_y columns to a Crop_startday XLSX by matching tile names.",
    )
    parser.add_argument("--xlsx", type=Path, required=True, help="Workbook to update in place.")
    parser.add_argument("--csv", type=Path, required=True, help="CSV containing tile radii.")
    return parser.parse_args()


def col_to_index(col: str) -> int:
    value = 0
    for ch in col:
        value = value * 26 + (ord(ch.upper()) - ord("A") + 1)
    return value


def index_to_col(index: int) -> str:
    chars = []
    while index > 0:
        index, rem = divmod(index - 1, 26)
        chars.append(chr(ord("A") + rem))
    return "".join(reversed(chars))


def split_cell_ref(cell_ref: str) -> Tuple[str, int]:
    col = "".join(ch for ch in cell_ref if ch.isalpha())
    row = int("".join(ch for ch in cell_ref if ch.isdigit()))
    return col, row


def get_cell_text(cell: ET.Element, shared_strings: List[str]) -> str:
    cell_type = cell.get("t")
    if cell_type == "inlineStr":
        is_node = cell.find(f"{{{SPREADSHEET_NS}}}is")
        if is_node is None:
            return ""
        return "".join(node.text or "" for node in is_node.iter(f"{{{SPREADSHEET_NS}}}t"))

    value = cell.find(f"{{{SPREADSHEET_NS}}}v")
    if value is None:
        return ""
    if cell_type == "s":
        return shared_strings[int(value.text)]
    return value.text or ""


def set_inline_string(cell: ET.Element, text: str) -> None:
    cell.attrib.pop("t", None)
    for child in list(cell):
        cell.remove(child)
    cell.set("t", "inlineStr")
    is_node = ET.SubElement(cell, f"{{{SPREADSHEET_NS}}}is")
    text_node = ET.SubElement(is_node, f"{{{SPREADSHEET_NS}}}t")
    text_node.text = text


def set_number(cell: ET.Element, number_text: str) -> None:
    cell.attrib.pop("t", None)
    for child in list(cell):
        cell.remove(child)
    value = ET.SubElement(cell, f"{{{SPREADSHEET_NS}}}v")
    value.text = number_text


def load_shared_strings(xlsx: Path) -> List[str]:
    with zipfile.ZipFile(xlsx) as zf:
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    return [
        "".join(node.text or "" for node in item.iter(f"{{{SPREADSHEET_NS}}}t"))
        for item in root.findall("ss:si", NS)
    ]


def load_radius_map(csv_path: Path) -> Dict[str, Tuple[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        radius_map = {}
        for row in reader:
            tile = row["tile"]
            radius_map[tile] = (row["radius_x_crs"], row["radius_y_crs"])
    return radius_map


def update_sheet(sheet_root: ET.Element, shared_strings: List[str], radius_map: Dict[str, Tuple[str, str]]) -> Tuple[int, int]:
    sheet_data = sheet_root.find("ss:sheetData", NS)
    if sheet_data is None:
        raise ValueError("Worksheet has no sheetData")

    rows = sheet_data.findall("ss:row", NS)
    if not rows:
        raise ValueError("Worksheet has no rows")

    header_row = rows[0]
    header_cells = header_row.findall("ss:c", NS)
    headers: dict[str, ET.Element] = {}
    max_col_index = 0
    for cell in header_cells:
        col, _ = split_cell_ref(cell.attrib["r"])
        headers[get_cell_text(cell, shared_strings)] = cell
        max_col_index = max(max_col_index, col_to_index(col))

    if "Tile" not in headers:
        raise ValueError("Could not find 'Tile' column in workbook header")

    radius_x_col = index_to_col(max_col_index + 1)
    radius_y_col = index_to_col(max_col_index + 2)

    radius_x_header = ET.Element(f"{{{SPREADSHEET_NS}}}c", {"r": f"{radius_x_col}1", "t": "inlineStr"})
    set_inline_string(radius_x_header, "radius_x")
    radius_y_header = ET.Element(f"{{{SPREADSHEET_NS}}}c", {"r": f"{radius_y_col}1", "t": "inlineStr"})
    set_inline_string(radius_y_header, "radius_y")
    header_row.append(radius_x_header)
    header_row.append(radius_y_header)

    tile_col, _ = split_cell_ref(headers["Tile"].attrib["r"])
    updated_rows = 0

    for row in rows[1:]:
        cells = row.findall("ss:c", NS)
        tile_value = ""
        for cell in cells:
            col, _ = split_cell_ref(cell.attrib["r"])
            if col == tile_col:
                tile_value = get_cell_text(cell, shared_strings)
                break

        if not tile_value or tile_value not in radius_map:
            continue

        radius_x_value, radius_y_value = radius_map[tile_value]
        row_index = row.attrib["r"]

        radius_x_cell = ET.Element(f"{{{SPREADSHEET_NS}}}c", {"r": f"{radius_x_col}{row_index}"})
        set_number(radius_x_cell, radius_x_value)
        radius_y_cell = ET.Element(f"{{{SPREADSHEET_NS}}}c", {"r": f"{radius_y_col}{row_index}"})
        set_number(radius_y_cell, radius_y_value)
        row.append(radius_x_cell)
        row.append(radius_y_cell)
        updated_rows += 1

    dimension = sheet_root.find("ss:dimension", NS)
    if dimension is not None:
        last_row_idx = int(rows[-1].attrib["r"])
        dimension.set("ref", f"A1:{radius_y_col}{last_row_idx}")

    return updated_rows, len(rows) - 1


def main() -> None:
    args = parse_args()
    shared_strings = load_shared_strings(args.xlsx)
    radius_map = load_radius_map(args.csv)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        with zipfile.ZipFile(args.xlsx, "r") as src_zip:
            src_zip.extractall(tmpdir_path)

        sheet_path = tmpdir_path / "xl" / "worksheets" / "sheet1.xml"
        sheet_root = ET.parse(sheet_path).getroot()
        updated_rows, data_rows = update_sheet(sheet_root, shared_strings, radius_map)
        ET.ElementTree(sheet_root).write(sheet_path, encoding="utf-8", xml_declaration=True)

        backup_path = args.xlsx.with_suffix(args.xlsx.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(args.xlsx, backup_path)

        with zipfile.ZipFile(args.xlsx, "w", compression=zipfile.ZIP_DEFLATED) as out_zip:
            for path in sorted(tmpdir_path.rglob("*")):
                if path.is_file():
                    out_zip.write(path, path.relative_to(tmpdir_path).as_posix())

    print(f"Updated {updated_rows} of {data_rows} workbook rows in {args.xlsx}")


if __name__ == "__main__":
    main()
