"""Profile warmed production inference without modifying framework sources."""
import argparse
import gzip
import json
import os
from pathlib import Path
import sys
import time

import torch
from profile_hook import event_snapshot
from profiler_reports import export_kernel_reports
from validate_reports import validate


def write_json(path, value):
    if int(os.environ.get('RANK', '0')) != 0 and path.name not in {'batch_shapes.json', 'resolved_samples.json'}:
        return
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + '\n')


def capture(root, fn):
    if int(os.environ.get('RANK', '0')) != 0:
        result = fn()
        torch.cuda.synchronize()
        torch.distributed.barrier()
        return result
    reports = root / 'reports'
    reports.mkdir(exist_ok=False)
    torch.cuda.synchronize()
    started = time.time()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True, profile_memory=False, with_stack=False,
    ) as prof:
        with torch.profiler.record_function('kernel_list.inference_request'):
            result = fn()
        torch.cuda.synchronize()
    print('CAPTURE_COMPLETE exporting', flush=True)
    trace = reports / 'rank-0.json.gz'
    prof.export_chrome_trace(str(trace))
    events = prof.events()
    with gzip.open(root / 'rank-0_report_events.json.gz', 'wt') as stream:
        json.dump(list(event_snapshot(events)), stream)
    export_kernel_reports(events, reports, 0, trace)
    quality = validate(reports)
    write_json(root / 'report_quality.json', quality)
    write_json(root / 'profile_window.json', {
        'warmup_requests': 1, 'active_requests': 1, 'rank': 0,
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(), 'elapsed_with_export_s': time.time() - started,
        'scope': 'complete warmed generation request; excludes checkpoint loading',
    })
    print('REPORT_COMPLETE', quality['unique_report_kernels'], quality['report_kernel_calls'], flush=True)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    return result


def t2v(root, argv):
    from cosmos_framework.scripts import inference as cli
    from cosmos_framework.inference.inference import OmniInference
    original = OmniInference.generate_batch
    calls = 0
    def wrapped(self, sample_args_list, data_batch, *, save_outputs=True):
        nonlocal calls
        calls += 1
        assert calls == 1, 'Expected exactly one batch'
        rank_root = root / 'batches' / f'rank-{os.environ.get("RANK", "0")}'
        rank_root.mkdir(parents=True, exist_ok=True)
        write_json(rank_root / 'resolved_samples.json', [s.model_dump(mode='json') for s in sample_args_list])
        write_json(rank_root / 'batch_shapes.json', {
            k: {'shape': list(v.shape), 'dtype': str(v.dtype)}
            for k, v in data_batch.items() if isinstance(v, torch.Tensor)
        })
        print('WARMUP_BEGIN', flush=True)
        original(self, sample_args_list, data_batch, save_outputs=False)
        torch.cuda.synchronize()
        print('WARMUP_COMPLETE', flush=True)
        return capture(root, lambda: original(self, sample_args_list, data_batch, save_outputs=save_outputs))
    OmniInference.generate_batch = wrapped
    sys.argv = ['inference', *argv]
    cli.main()
    assert calls == 1


def vlm(root, argv):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args(argv)
    specification = json.loads(args.input.read_text())
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, device_map='cuda:0',
    ).eval()
    processor = AutoProcessor.from_pretrained(args.checkpoint)
    processor.tokenizer.padding_side = 'left'
    inputs = processor.apply_chat_template(
        specification['messages'], add_generation_prompt=True, tokenize=True,
        return_tensors='pt', return_dict=True,
    ).to('cuda:0')
    write_json(root / 'batch_shapes.json', {
        k: {'shape': list(v.shape), 'dtype': str(v.dtype),
            **({'values': v.tolist()} if 'grid' in k else {})}
        for k, v in inputs.items() if isinstance(v, torch.Tensor)
    })
    write_json(root / 'model_runtime.json', {
        'class': type(model).__name__, 'checkpoint': args.checkpoint,
        'attention_implementation': model.config._attn_implementation,
        'text_attention': model.config.text_config._attn_implementation,
        'vision_attention': model.config.vision_config._attn_implementation,
        'parameters': sum(p.numel() for p in model.parameters()),
        'dtype': str(model.dtype), 'batch_size': 1,
        'max_new_tokens': specification['max_new_tokens'],
        'do_sample': False, 'use_cache': model.generation_config.use_cache,
    })
    phase_calls = {'prefill': 0, 'decode': 0, 'vision': 0}
    def wrap_forward(module, phase_fn):
        original = module.forward
        def forward(*args, **kwargs):
            phase = phase_fn(kwargs)
            phase_calls[phase] += 1
            with torch.profiler.record_function('kernel_list.' + phase):
                return original(*args, **kwargs)
        module.forward = forward
    wrap_forward(model, lambda kw: 'prefill' if kw['input_ids'].shape[1] > 1 else 'decode')
    wrap_forward(model.model.visual, lambda kw: 'vision')
    def generate():
        with torch.inference_mode():
            return model.generate(**inputs, max_new_tokens=specification['max_new_tokens'], do_sample=False)
    torch.manual_seed(42)
    print('WARMUP_BEGIN', flush=True)
    warm = generate()
    torch.cuda.synchronize()
    print('WARMUP_COMPLETE', flush=True)
    for k in phase_calls:
        phase_calls[k] = 0
    output = capture(root, generate)
    generated = output[:, inputs['input_ids'].shape[-1]:]
    write_json(root / 'generation.json', {
        'response': processor.batch_decode(generated, skip_special_tokens=True),
        'input_tokens': inputs['input_ids'].shape[-1], 'generated_tokens': generated.shape[-1],
        'warmup_matches_capture': torch.equal(warm, output), 'phase_calls': phase_calls,
    })
    assert phase_calls['vision'] and phase_calls['prefill'] and phase_calls['decode']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['t2v', 'vlm_only'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args, rest = parser.parse_known_args()
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'runtime.json', {
        'argv': sys.argv, 'python': sys.executable, 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
    })
    (t2v if args.mode == 't2v' else vlm)(args.output, rest)
    write_json(args.output / 'completion.json', {'status': 'complete', 'mode': args.mode})


if __name__ == '__main__':
    main()
