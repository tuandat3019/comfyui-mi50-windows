"""Submit an API workflow to a local ComfyUI server and retain its result/timing."""
from pathlib import Path
import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid


def request(url, value=None, data=None, content_type=None):
    if value is not None:
        data = json.dumps(value).encode('utf-8')
        content_type = 'application/json; charset=utf-8'
    headers = {'Content-Type': content_type} if content_type else {}
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers), timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f'ComfyUI HTTP {exc.code}: {exc.read().decode("utf-8", errors="replace")}') from exc


def upload(url, image):
    image = Path(image)
    boundary = 'mi50-' + uuid.uuid4().hex
    # Use a generated filename, avoiding client filesystem paths in workflow metadata.
    name = 'reference-' + uuid.uuid4().hex + image.suffix.lower()
    data = (f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\n'
            'Content-Type: application/octet-stream\r\n\r\n').encode() + image.read_bytes()
    data += f'\r\n--{boundary}--\r\n'.encode()
    result = request(url + '/upload/image', data=data, content_type='multipart/form-data; boundary=' + boundary)
    return '/'.join(part for part in (result.get('subfolder'), result['name']) if part)


def cancel_own_prompt(url, prompt_id):
    queue = request(url + '/queue')
    if any(item[1] == prompt_id for item in queue.get('queue_running', [])):
        # ComfyUI's interrupt endpoint is global. A queue/read race could
        # interrupt another client's job, so leave the running task intact.
        print('[cancel] This prompt is already running. Stop its dedicated launcher with Ctrl+C.', flush=True)
    elif any(item[1] == prompt_id for item in queue.get('queue_pending', [])):
        request(url + '/queue', value={'delete': [prompt_id]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workflow', required=True)
    parser.add_argument('--port', type=int, default=8188)
    parser.add_argument('--image', help='An already prepared PNG; for TRELLIS use a complete object with transparent background')
    parser.add_argument('--prompt', help='Override the Z-Image text prompt')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--timeout', type=float, default=3600, help='Seconds; cold ZLUDA compilation can be slow')
    parser.add_argument('--out-dir', default='outputs/request')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.timeout <= 0:
        parser.error('Invalid port or timeout')
    url = f'http://127.0.0.1:{args.port}'
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=False)
    graph = json.loads(Path(args.workflow).read_text(encoding='utf-8'))
    if args.image:
        name = upload(url, args.image)
        loaders = [node for node in graph.values() if node['class_type'] == 'Trellis2LoadImageWithTransparency']
        if len(loaders) != 1:
            raise RuntimeError('--image requires one TRELLIS image loader')
        loaders[0]['inputs']['image'] = name
    if args.prompt is not None:
        encoders = [node for node in graph.values() if node['class_type'] == 'CLIPTextEncode']
        if len(encoders) != 1:
            raise RuntimeError('--prompt requires one Z-Image text encoder')
        encoders[0]['inputs']['text'] = args.prompt
    if args.seed is not None:
        for node in graph.values():
            if 'seed' in node['inputs']:
                node['inputs']['seed'] = args.seed
    info = request(url + '/object_info')
    missing = sorted({node['class_type'] for node in graph.values()} - info.keys())
    if missing:
        raise RuntimeError('Missing workflow nodes: ' + ', '.join(missing))
    (output / 'workflow-api.json').write_text(json.dumps(graph, indent=2) + '\n', encoding='utf-8')
    started = time.monotonic()
    result = request(url + '/prompt', value={'prompt': graph, 'client_id': uuid.uuid4().hex})
    if result.get('node_errors') or 'prompt_id' not in result:
        raise RuntimeError(f'ComfyUI rejected the workflow: {result}')
    prompt_id = result['prompt_id']
    print('PROMPT_ID', prompt_id, flush=True)
    try:
        while time.monotonic() - started < args.timeout:
            history = request(url + '/history/' + urllib.parse.quote(prompt_id))
            if prompt_id in history:
                entry = history[prompt_id]
                timing = {'request_wall_s': round(time.monotonic() - started, 3), 'prompt_id': prompt_id,
                          'includes_queue_and_model_load': True, 'includes_server_start': False}
                (output / 'history.json').write_text(json.dumps(entry, indent=2) + '\n', encoding='utf-8')
                (output / 'timing.json').write_text(json.dumps(timing, indent=2) + '\n', encoding='utf-8')
                if entry.get('status', {}).get('status_str') != 'success':
                    raise RuntimeError('Workflow failed; inspect history.json and the ComfyUI console.')
                print(json.dumps(timing, indent=2))
                print('[result] Artifacts are in the ComfyUI runtime output directory; file references are in history.json.')
                return
            time.sleep(1)
        raise TimeoutError('Workflow timed out; cold compile may require a longer --timeout.')
    except (KeyboardInterrupt, TimeoutError):
        cancel_own_prompt(url, prompt_id)
        raise


if __name__ == '__main__':
    main()
