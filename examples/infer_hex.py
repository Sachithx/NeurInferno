"""Cut fields in a few hex messages using weights from Hugging Face."""

from neurinferno import FieldBoundaryModel

HEX = [
    "0001080006040001900c2d9bfa4649e7160700000000000043f03612",
    "0001080006040002fbccad5c9fb1d014735252376fd2446375217d01",
    "0001080006040002a388be2d1d684df0f31908ae1acad25b4dca7aa8",
    "0001080006040001d41cd6668e87ac1e6d7b0000000000008593a2d9",
]

model = FieldBoundaryModel.from_pretrained()  # sachithabey/neurinferno
for i, result in enumerate(model.infer(HEX, threshold=0.75), start=1):
    print(f"message {i} ({result.n_bytes} bytes)")
    for seg in result.segments:
        print(f"  [{seg.start}:{seg.end}] {seg.hex}")
