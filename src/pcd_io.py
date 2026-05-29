from pathlib import Path
import struct
import numpy as np


def _parse_header(path):
    header = {}
    lines = []
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"PCD header has no DATA line: {path}")
            text = line.decode("ascii", errors="ignore").strip()
            lines.append(line)
            if text:
                parts = text.split()
                header[parts[0].upper()] = parts[1:]
            if text.upper().startswith("DATA"):
                break
        data_offset = f.tell()
    return header, data_offset


def read_pcd_xyz(path):
    """Read an ASCII or binary PCD and return Nx3 float32 xyz points."""
    path = Path(path)
    header, data_offset = _parse_header(path)
    fields = header.get("FIELDS", [])
    sizes = [int(x) for x in header.get("SIZE", [])]
    types = header.get("TYPE", [])
    counts = [int(x) for x in header.get("COUNT", ["1"] * len(fields))]
    points = int(header.get("POINTS", header.get("WIDTH", ["0"]))[0])
    data = header.get("DATA", [""])[0].lower()
    if fields[:3] != ["x", "y", "z"]:
        raise ValueError(f"Expected x y z fields at the front of {path}, got {fields}")

    if data == "ascii":
        arr = np.loadtxt(path, comments="#", skiprows=len(open(path, "rb").read()[:data_offset].splitlines()))
        return np.asarray(arr[:, :3], dtype=np.float32)

    dtype_fields = []
    scalar_dtypes = []
    for field, size, typ, count in zip(fields, sizes, types, counts):
        dt = _numpy_dtype(typ, size, field)
        scalar_dtypes.append(dt)
        dtype_fields.append((field, dt, count) if count > 1 else (field, dt))

    if data == "binary_compressed":
        with open(path, "rb") as f:
            f.seek(data_offset)
            compressed_size, uncompressed_size = struct.unpack("<II", f.read(8))
            compressed = f.read(compressed_size)
        raw = _lzf_decompress(compressed, uncompressed_size)
        columns = {}
        offset = 0
        for field, size, count, dt in zip(fields, sizes, counts, scalar_dtypes):
            nbytes = points * size * count
            buf = memoryview(raw)[offset : offset + nbytes]
            arr = np.frombuffer(buf, dtype=dt, count=points * count)
            if count > 1:
                arr = arr.reshape(points, count)
            columns[field] = arr
            offset += nbytes
        xyz = np.column_stack([columns["x"], columns["y"], columns["z"]]).astype(np.float32, copy=False)
        return xyz[np.isfinite(xyz).all(axis=1)]

    if data != "binary":
        raise ValueError(f"Unsupported PCD DATA mode {data!r}: {path}")

    dtype = np.dtype(dtype_fields)
    with open(path, "rb") as f:
        f.seek(data_offset)
        cloud = np.frombuffer(f.read(points * dtype.itemsize), dtype=dtype, count=points)
    xyz = np.column_stack([cloud["x"], cloud["y"], cloud["z"]]).astype(np.float32, copy=False)
    return xyz[np.isfinite(xyz).all(axis=1)]


def _numpy_dtype(typ, size, field):
    if typ == "F" and size == 4:
        return np.float32
    if typ == "F" and size == 8:
        return np.float64
    if typ == "U" and size == 4:
        return np.uint32
    if typ == "I" and size == 4:
        return np.int32
    if typ == "U" and size == 2:
        return np.uint16
    if typ == "I" and size == 2:
        return np.int16
    if typ == "U" and size == 1:
        return np.uint8
    if typ == "I" and size == 1:
        return np.int8
    raise ValueError(f"Unsupported PCD field type {field}: {typ}{size}")


def _lzf_decompress(data, expected_size):
    """Decompress the LZF block used by PCL binary_compressed PCD files."""
    out = bytearray(expected_size)
    ip = 0
    op = 0
    data_len = len(data)
    while ip < data_len:
        ctrl = data[ip]
        ip += 1
        if ctrl < 32:
            length = ctrl + 1
            out[op : op + length] = data[ip : ip + length]
            ip += length
            op += length
        else:
            length = ctrl >> 5
            ref = op - ((ctrl & 0x1F) << 8) - 1
            if length == 7:
                length += data[ip]
                ip += 1
            ref -= data[ip]
            ip += 1
            length += 2
            for _ in range(length):
                out[op] = out[ref]
                op += 1
                ref += 1
    if op != expected_size:
        raise ValueError(f"LZF decompressed size mismatch: expected {expected_size}, got {op}")
    return bytes(out)
