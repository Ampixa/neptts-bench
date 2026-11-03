"""Hash-verified CPU Zipformer decoding; no service sockets or private call inputs."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_online(export, threads):
    import sherpa_onnx
    names = ['encoder.int8.onnx', 'decoder.int8.onnx', 'joiner.int8.onnx', 'tokens.txt', 'export_meta.json']
    metadata = json.loads((export/'export_meta.json').read_text())
    print('SCHEMA export', sorted(metadata), flush=True)
    for name, item in metadata.get('files', {}).items():
        if name in names and sha(export/name) != item['sha256']:
            raise RuntimeError('Export hash mismatch: '+name)
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(export/'tokens.txt'), encoder=str(export/'encoder.int8.onnx'),
        decoder=str(export/'decoder.int8.onnx'), joiner=str(export/'joiner.int8.onnx'),
        num_threads=threads, sample_rate=16000, feature_dim=80, low_freq=20., high_freq=-400.,
        dither=0., normalize_samples=True, snip_edges=False, enable_endpoint_detection=False,
        decoding_method='greedy_search', model_type='zipformer2', provider='cpu', modeling_unit='bpe')

    def decode(samples):
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, samples)
        stream.accept_waveform(16000, np.zeros(12800, dtype=np.float32))
        stream.input_finished()
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        return recognizer.get_result(stream)
    return decode, dict(export=str(export), hashes={n: sha(export/n) for n in names},
                        sherpa_onnx=sherpa_onnx.__version__, padding_samples=12800,
                        export_metadata=metadata)


def build_offline(args):
    import torch
    torch.set_num_threads(args.threads)
    import sentencepiece as spm
    from lhotse import Fbank, FbankConfig
    sys.path[:0] = [str(args.recipe), str(args.icefall)]
    from train import add_model_arguments, get_model, get_params
    from beam_search import greedy_search_batch
    # The frozen training module initializes interop threads at import time.
    torch.set_num_threads(args.threads)
    if sha(args.checkpoint) != '97392051e0fd5aa764b0098c7a39ebe2223e521c76701bfffdf5b99f1f39b213':
        raise RuntimeError('D32 checkpoint pin mismatch')
    if sha(args.bpe) != 'efef2f02304881a9a5aa269aec829b9ee1ba1ada0ff48f1f2d3feae7fb51394f':
        raise RuntimeError('Tokenizer pin mismatch')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    print('SCHEMA checkpoint', sorted(checkpoint), 'args', sorted(checkpoint['args']), flush=True)
    params = get_params()
    # Export metadata stores command-line values as strings. Apply the frozen
    # recipe's own argument converters instead of passing strings as dimensions.
    model_parser = argparse.ArgumentParser()
    add_model_arguments(model_parser)
    model_parser.add_argument('--context-size', type=int, default=2)
    defaults = vars(model_parser.parse_args([]))
    params.update(defaults)
    for action in model_parser._actions:
        if action.dest in checkpoint['args']:
            value = checkpoint['args'][action.dest]
            params[action.dest] = action.type(value) if action.type and isinstance(value, str) else value
    assert '-1' in str(params.chunk_size).split(',')
    assert '-1' in str(params.left_context_frames).split(',')
    params.chunk_size = '-1'
    params.left_context_frames = '-1'
    sp = spm.SentencePieceProcessor(model_file=str(args.bpe))
    params.vocab_size = sp.get_piece_size()
    params.blank_id = sp.piece_to_id('<blk>')
    params.unk_id = sp.piece_to_id('<unk>')
    params.sos_id = params.eos_id = sp.piece_to_id('<sos/eos>')
    assert params.blank_id == 0
    model = get_model(params).eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    features = Fbank(FbankConfig(num_filters=80))

    @torch.no_grad()
    def decode(samples):
        f = torch.from_numpy(features.extract(samples.reshape(1, -1), 16000)).unsqueeze(0)
        lens = torch.tensor([f.shape[1]+30], dtype=torch.int64)
        f = torch.nn.functional.pad(f, (0, 0, 0, 30), value=math.log(1e-10))
        out, out_lens = model.forward_encoder(f, lens)
        tokens = greedy_search_batch(model=model, encoder_out=out, encoder_out_lens=out_lens)
        return ' '.join(sp.decode(tokens)[0].split())
    modules = ['train.py', 'zipformer.py', 'subsampling.py', 'model.py', 'beam_search.py']
    return decode, dict(checkpoint_sha256=sha(args.checkpoint), bpe_sha256=sha(args.bpe),
                        checkpoint=str(args.checkpoint), tokenizer=str(args.bpe),
                        chunk_size=-1, left_context_frames=-1, parameters=sum(p.numel() for p in model.parameters()),
                        features='lhotse Fbank(FbankConfig(num_filters=80)); 30 log-eps right-pad frames',
                        recipe=str(args.recipe), recipe_hashes={n: sha(args.recipe/n) for n in modules},
                        torch=torch.__version__, lhotse=importlib.metadata.version('lhotse'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mode', choices=['online', 'offline'], required=True)
    parser.add_argument('--export', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--bpe', type=Path)
    parser.add_argument('--recipe', type=Path)
    parser.add_argument('--icefall', type=Path)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--input-root', type=Path)
    args = parser.parse_args()
    tasks = json.loads(args.tasks.read_text())
    print('SCHEMA task', sorted(tasks[0]), flush=True)
    if args.out.exists():
        raise FileExistsError(args.out)
    decode, meta = build_online(args.export, args.threads) if args.mode == 'online' else build_offline(args)
    meta.update(mode=args.mode, threads=args.threads, provider='cpu', task_sha256=sha(args.tasks),
                worker_sha256=sha(Path(__file__)))
    args.out.with_suffix('.meta.json').write_text(json.dumps(meta, indent=2))
    with args.out.open('x') as output:
        for i, task in enumerate(tasks):
            start = time.perf_counter()
            path = Path(task['audio'])
            if {'eval_private', 'khalti', 'calls', 'orbit'} & set(path.parts):
                raise RuntimeError('Forbidden input path')
            if args.input_root:
                path = args.input_root/str(path).lstrip('/')
            if sha(path) != task['sha256']:
                raise RuntimeError('Input hash mismatch')
            samples, sr = sf.read(path, dtype='float32', always_2d=True)
            samples = samples.mean(axis=1)
            if sr != 16000:
                g = math.gcd(sr, 16000)
                samples = resample_poly(samples, 16000//g, sr//g).astype(np.float32)
            hyp = decode(samples)
            output.write(json.dumps(dict(task, hyp=hyp, status='ok', seconds=time.perf_counter()-start),
                                    ensure_ascii=False)+'\n')
            output.flush()
            print(i+1, len(tasks), flush=True)


if __name__ == '__main__':
    main()
