#!/usr/bin/env python3
"""Write a g4lab-expert-manifest-v2 manifest for a Gemma 4 MoE GGUF.

The gemma4 engine reads routed experts either from per-layer packed files or,
when a layer's packed file is absent, straight from the fused GGUF tensors
("direct GGUF fallback"). This tool describes every routed layer of a GGUF in
the second form, so no separate packing step is needed:

    python3 tools/make_gemma4_manifest.py MODEL.gguf OUT_DIR
    ./gemma4 generate MODEL.gguf OUT_DIR "Hello" --max-new 16

The manifest stores the GGUF path relative to OUT_DIR (e.g. ../model.gguf) when it can.

Only the GGUF header and tensor directory are read. The expert tensors must be
Q4_0 (the engine's only routed-expert format), laid out as the fused
`ffn_gate_up_exps` ([model_width, 2 * expert_width, experts]) plus
`ffn_down_exps` ([expert_width, model_width, experts]).
"""
import argparse
import json
import os
import struct
import sys
from pathlib import Path

SCALARS = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
T_STRING, T_ARRAY = 8, 9
GGML_F32, GGML_Q4_0 = 0, 2
Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMS = 18, 32


class Reader:
    def __init__(self, handle):
        self.f = handle

    def take(self, fmt):
        size = struct.calcsize(fmt)
        data = self.f.read(size)
        if len(data) != size:
            raise ValueError("truncated GGUF")
        return struct.unpack("<" + fmt, data)[0]

    def string(self):
        n = self.take("Q")
        data = self.f.read(n)
        if len(data) != n:
            raise ValueError("truncated GGUF string")
        return data.decode("utf-8", "replace")

    def skip_value(self, vtype):
        if vtype in SCALARS:
            self.f.seek(SCALARS[vtype], 1)
        elif vtype == T_STRING:
            self.f.seek(self.take("Q"), 1)
        elif vtype == T_ARRAY:
            etype, count = self.take("I"), self.take("Q")
            if etype in SCALARS:
                self.f.seek(SCALARS[etype] * count, 1)
            else:
                for _ in range(count):
                    self.skip_value(etype)
        else:
            raise ValueError(f"unsupported GGUF value type {vtype}")


def read_directory(path):
    """Return (kv, tensors, data_start). Only the keys the tool needs are kept."""
    wanted = {"general.architecture", "general.alignment", "gemma4.expert_count",
              "gemma4.expert_used_count", "gemma4.embedding_length",
              "gemma4.expert_feed_forward_length", "gemma4.block_count"}
    kv = {}
    with open(path, "rb") as handle:
        r = Reader(handle)
        if handle.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        version = r.take("I")
        if version not in (2, 3):
            raise ValueError(f"unsupported GGUF version {version}")
        n_tensors, n_kv = r.take("Q"), r.take("Q")
        for _ in range(n_kv):
            key, vtype = r.string(), r.take("I")
            if key in wanted and (vtype in SCALARS or vtype == T_STRING):
                if vtype == T_STRING:
                    kv[key] = r.string()
                else:
                    fmt = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
                           7: "B", 10: "Q", 11: "q", 12: "d"}[vtype]
                    kv[key] = r.take(fmt)
            else:
                r.skip_value(vtype)
        tensors = {}
        for _ in range(n_tensors):
            name, n_dims = r.string(), r.take("I")
            dims = [r.take("Q") for _ in range(n_dims)]
            ttype, offset = r.take("I"), r.take("Q")
            tensors[name] = (dims, ttype, offset)
        align = int(kv.get("general.alignment", 32))
        data_start = (handle.tell() + align - 1) // align * align
    return kv, tensors, data_start


def source_reference(model, out_dir):
    """Path the engine uses to find the GGUF. Relative to the manifest directory when
    both live on one filesystem tree (the usual `model.gguf` next to `packed/`), so the
    model directory can be mounted anywhere, e.g. into a container; otherwise absolute.
    The model path is NOT resolved: a symlinked model.gguf stays a stable name."""
    model_abs = os.path.abspath(model)
    if out_dir is None:
        return model_abs
    try:
        return os.path.relpath(model_abs, os.path.abspath(out_dir))
    except ValueError:      # different drives on Windows
        return model_abs


def build(model, out_dir=None):
    kv, tensors, data_start = read_directory(model)
    if kv.get("general.architecture") != "gemma4":
        raise ValueError(f"architecture is {kv.get('general.architecture')!r}, not gemma4")
    layers_total = int(kv["gemma4.block_count"])
    experts = int(kv["gemma4.expert_count"])
    used = int(kv["gemma4.expert_used_count"])
    layers = []
    for layer in range(layers_total):
        gate_up = tensors.get(f"blk.{layer}.ffn_gate_up_exps.weight")
        down = tensors.get(f"blk.{layer}.ffn_down_exps.weight")
        scale = tensors.get(f"blk.{layer}.ffn_down_exps.scale")
        if gate_up is None or down is None:
            raise ValueError(f"layer {layer} has no fused routed-expert tensors")
        (width, fused_rows, n_exp), gu_type, gu_off = gate_up
        (expert_width, down_width, down_n_exp), down_type, down_off = down
        if gu_type != GGML_Q4_0 or down_type != GGML_Q4_0:
            raise ValueError(f"layer {layer} experts are not Q4_0 "
                             f"(types {gu_type}/{down_type}); quantize to Q4_0 first")
        if n_exp != experts or down_n_exp != experts or fused_rows != 2 * expert_width \
                or down_width != width:
            raise ValueError(f"layer {layer} expert tensor shapes are inconsistent")
        if width % Q4_0_BLOCK_ELEMS or expert_width % Q4_0_BLOCK_ELEMS:
            raise ValueError("expert dimensions must be multiples of 32")
        gate_bytes = expert_width * (width // Q4_0_BLOCK_ELEMS) * Q4_0_BLOCK_BYTES
        down_bytes = width * (expert_width // Q4_0_BLOCK_ELEMS) * Q4_0_BLOCK_BYTES
        payload = 2 * gate_bytes + down_bytes
        entry = {
            "layer": layer, "expert_count": experts, "model_width": width,
            "expert_width": expert_width, "source_layout": "fused_gate_up",
            "packed_file": f"layer-{layer:02d}.g4ex", "payload_bytes": payload,
            "record_stride": (payload + 4095) // 4096 * 4096,
            "expert_scale": None,
            "components": [
                {"role": "gate", "tensor": f"blk.{layer}.ffn_gate_up_exps.weight",
                 "ggml_type": GGML_Q4_0, "type_name": "Q4_0", "slice_bytes": gate_bytes,
                 "source_offset": data_start + gu_off,
                 "source_expert_stride": 2 * gate_bytes, "source_within_expert_offset": 0},
                {"role": "up", "tensor": f"blk.{layer}.ffn_gate_up_exps.weight",
                 "ggml_type": GGML_Q4_0, "type_name": "Q4_0", "slice_bytes": gate_bytes,
                 "source_offset": data_start + gu_off,
                 "source_expert_stride": 2 * gate_bytes,
                 "source_within_expert_offset": gate_bytes},
                {"role": "down", "tensor": f"blk.{layer}.ffn_down_exps.weight",
                 "ggml_type": GGML_Q4_0, "type_name": "Q4_0", "slice_bytes": down_bytes,
                 "source_offset": data_start + down_off,
                 "source_expert_stride": down_bytes, "source_within_expert_offset": 0},
            ],
        }
        if scale is not None:
            if scale[1] != GGML_F32 or scale[0] != [experts]:
                raise ValueError(f"layer {layer} expert scale must be F32 [{experts}]")
            entry["expert_scale"] = {"ggml_type": GGML_F32, "source_offset": data_start + scale[2],
                                     "scalar_bytes": 4}
        layers.append(entry)
    return {"format": "g4lab-expert-manifest-v2", "architecture": "gemma4",
            "source": source_reference(model, out_dir), "expert_count": experts,
            "expert_used_count": used, "layers": layers}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("model", type=Path)
    parser.add_argument("out", type=Path, help="directory that will hold manifest.json")
    args = parser.parse_args()
    try:
        manifest = build(args.model, args.out)
    except (OSError, ValueError, KeyError) as error:
        sys.exit(f"error: {error}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {args.out / 'manifest.json'}: {len(manifest['layers'])} layers, "
          f"{manifest['expert_count']} experts, top-{manifest['expert_used_count']}")


if __name__ == "__main__":
    main()
