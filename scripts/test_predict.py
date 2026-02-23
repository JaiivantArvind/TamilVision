"""One-shot test: sends a real BMP to /predict and prints the response."""
import json
import pathlib
import urllib.request

IMG = pathlib.Path(
    r"D:\VIT\SEM-6\Projects\SC\T-2\data\raw"
    r"\uTHCD_b(80-20-split)\80-20-split\train-test-classwise\train"
    r"\ர\0004_081.bmp"
)

boundary = "----FormBoundaryTest1234"
crlf = b"\r\n"

body = (
    ("--" + boundary + "\r\n").encode()
    + b"Content-Disposition: form-data; name=\"file\"; filename=\"char.bmp\"\r\n"
    + b"Content-Type: image/bmp\r\n\r\n"
    + IMG.read_bytes()
    + ("\r\n--" + boundary + "--\r\n").encode()
)

req = urllib.request.Request(
    "http://localhost:8000/predict",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
)
resp = urllib.request.urlopen(req)
print(json.dumps(json.loads(resp.read()), ensure_ascii=False, indent=2))
