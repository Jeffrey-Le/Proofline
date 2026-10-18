"""Check one posting from the terminal:  python -m scripts.check path/to/posting.txt"""
import json
import sys

from proofline.pipeline import check_posting_sync

report, _ = check_posting_sync(open(sys.argv[1]).read())
print(json.dumps(report.model_dump(), indent=2))
