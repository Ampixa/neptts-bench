"""Cold-process numeric test-retest for21/205 prompts in every submission."""
import json
from pathlib import Path
import tempfile
import zipfile

from neptts_eval import d32
from submission_app.contract import SENTENCES
from calibrate_k2 import config, ROOT


def main():
    ids = sorted(SENTENCES)[::10]
    if len(ids) != 21:
        raise RuntimeError('Retest coverage changed')
    score = d32.load_normalizer(ROOT)
    decode = d32.build_decoder(config())
    inventory = json.loads((ROOT / 'approved_inventory.json').read_text())
    rows = []
    for item in inventory['approved']:
        sid = item['id']
        report = json.loads((ROOT / ('result-' + sid + '.json')).read_text())['report']
        changed = 0
        with zipfile.ZipFile(ROOT / ('package-' + sid + '.zip')) as archive:
            for prompt in ids:
                with tempfile.TemporaryDirectory(dir=ROOT / 'private_tmp') as temp:
                    path = Path(temp) / (prompt + '.wav')
                    data = archive.read('audio/' + prompt + '.wav')
                    path.write_bytes(data)
                    # Read-only, same byte hash verified by the package's first pass.
                    result = score(SENTENCES[prompt]['text_devanagari'], decode(d32.read_audio(path)))
                    changed += result != report['per_sentence'][prompt]
        rows.append(dict(submission_id=sid, n=21, total=205, changed_numeric_rows=changed))
        print('Retest ID', sid, 'changed', changed, flush=True)
    if any((ROOT / 'private_tmp').iterdir()):
        raise RuntimeError('Retest left extracted audio behind')
    output = dict(cold_process=True, scope='numeric CER/WER/edit counts; no stored private hypotheses',
                  rows=rows, passed=all(r['changed_numeric_rows'] == 0 for r in rows))
    (ROOT / 'submission_retest.json').write_text(json.dumps(output, indent=2) + '\n')
    if not output['passed']:
        raise RuntimeError('Submission numeric retest changed')


if __name__ == '__main__':
    main()
