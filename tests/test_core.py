from pathlib import Path
import hashlib
import importlib.util
import io
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mi50.downloads import extract_zip, verify
from mi50.paths import launch_environment


def script(name):
    spec = importlib.util.spec_from_file_location('mi50_test_' + name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ArchiveTests(unittest.TestCase):
    def archive(self, directory, entries):
        archive = Path(directory) / 'test.zip'
        with zipfile.ZipFile(archive, 'w') as bundle:
            for name, value in entries:
                # ZipInfo's constructor normalizes Windows separators. Set the
                # filename afterwards to exercise malformed archive members.
                entry = zipfile.ZipInfo('placeholder')
                entry.filename = name
                bundle.writestr(entry, value)
        return archive

    def test_traversal_rejected_before_any_file_is_written(self):
        for bad in ('../escape', '/escape', 'root/../../escape', 'C:/escape', 'root\\escape'):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as directory:
                archive = self.archive(directory, [('valid.txt', 'valid'), (bad, 'bad')])
                destination = Path(directory) / 'target'
                with self.assertRaises(ValueError):
                    extract_zip(archive, destination)
                self.assertFalse(destination.exists())

    def test_headers_only_and_strip_root(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = self.archive(directory, [('pkg/include/test.h', 'header'), ('pkg/lib/test.lib', 'library'), ('pkg/bin/test.dll', 'dll')])
            destination = Path(directory) / 'target'
            extract_zip(archive, destination, strip_root=True, headers_only=True)
            self.assertEqual((destination / 'include/test.h').read_text(), 'header')
            self.assertTrue((destination / 'lib/test.lib').is_file())
            self.assertFalse((destination / 'bin').exists())

    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / 'symlink.zip'
            with zipfile.ZipFile(archive, 'w') as bundle:
                entry = zipfile.ZipInfo('link')
                entry.create_system = 3
                entry.external_attr = 0o120777 << 16
                bundle.writestr(entry, '../outside')
            with self.assertRaises(ValueError):
                extract_zip(archive, Path(directory) / 'target')

    def test_checksum_and_size_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'archive'
            path.write_bytes(b'hello')
            record = {'bytes': 5, 'sha256': hashlib.sha256(b'hello').hexdigest()}
            verify(path, record)
            for wrong in ({**record, 'bytes': 4}, {**record, 'sha256': '0' * 64}):
                with self.assertRaises(RuntimeError):
                    verify(path, wrong)


class PackagingTests(unittest.TestCase):
    def test_all_python_sources_compile(self):
        for path in [*ROOT.glob('mi50/*.py'), *ROOT.glob('scripts/*.py')]:
            with self.subTest(path=path.name):
                compile(path.read_text(encoding='utf-8'), str(path), 'exec')

    def test_launch_selects_hip_gpu_and_profile_without_shared_python_overlay(self):
        launch = script('launch')
        runtime = Path(tempfile.gettempdir()) / 'runtime with spaces'
        with patch.dict(os.environ, {'PYTHONPATH': 'foreign-overlay'}):
            image, env = launch.command_for(runtime, 'image', 2, 8188)
        self.assertEqual(env['HIP_VISIBLE_DEVICES'], '2')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '0')
        self.assertNotIn('PYTHONPATH', env)
        self.assertEqual(image[image.index('--whitelist-custom-nodes') + 1], 'ComfyUI-GGUF')
        self.assertIn('--fp32-vae', image)
        self.assertEqual(image[0], str(runtime / 'zluda/zluda.exe'))
        trellis, _ = launch.command_for(runtime, 'trellis', 0, 8189)
        self.assertEqual(trellis[trellis.index('--whitelist-custom-nodes') + 1], 'ComfyUI-Trellis2-AMD')
        self.assertNotIn('--fp32-vae', trellis)

    def test_read_only_plans_work_without_runtime(self):
        for name, args in [('setup', ['--profile', 'all', '--plan']), ('launch', ['--profile', 'trellis', '--dry-run']),
                           ('download_models', ['--profile', 'image', '--plan'])]:
            with tempfile.TemporaryDirectory() as directory:
                runtime = Path(directory) / 'absent-runtime'
                result = subprocess.run([sys.executable, str(ROOT / 'scripts' / (name + '.py')), *args, '--runtime', str(runtime)],
                                        capture_output=True, text=True, check=True)
                self.assertIsInstance(json.loads(result.stdout), dict)
                self.assertFalse(runtime.exists())

    def test_pinned_manifests_and_artifact_hashes(self):
        sources = json.loads((ROOT / 'config/sources.json').read_text())
        for value in sources.values():
            self.assertRegex(value['commit'], r'^[0-9a-f]{40}$')
            self.assertTrue(value['url'].startswith('https://'))
        downloads = json.loads((ROOT / 'config/downloads.json').read_text())
        self.assertEqual(len(downloads), len({value['name'] for value in downloads}))
        for value in downloads:
            self.assertRegex(value['sha256'], r'^[0-9a-f]{64}$')
            self.assertGreater(value['bytes'], 0)
        results = json.loads((ROOT / 'benchmarks/results.json').read_text())
        for value in results['artifacts']:
            verify(ROOT / value['file'], value)
        self.assertFalse(results['standalone_clean_install_gpu_retested'])

    def test_api_workflows_have_valid_references_and_benchmark_settings(self):
        for filename in ('profiles/z-image/workflow-api.json', 'profiles/trellis2/workflow-api.json'):
            graph = json.loads((ROOT / filename).read_text())
            for node in graph.values():
                for value in node['inputs'].values():
                    if isinstance(value, list):
                        self.assertEqual(len(value), 2)
                        self.assertIn(value[0], graph)
                        self.assertIsInstance(value[1], int)
        image = json.loads((ROOT / 'profiles/z-image/workflow-api.json').read_text())
        self.assertEqual(image['7']['inputs']['steps'], 8)
        self.assertEqual(image['8']['class_type'], 'VAEDecodeTiled')
        trellis = json.loads((ROOT / 'profiles/trellis2/workflow-api.json').read_text())
        self.assertEqual(trellis['4']['inputs']['pipeline_type'], '1024')
        self.assertEqual(trellis['3']['inputs']['sparse_backend'], 'sdpa')

    def test_example_glb_header_and_face_count(self):
        data = (ROOT / 'examples/robot.glb').read_bytes()
        magic, version, length = struct.unpack_from('<III', data)
        self.assertEqual((magic, version, length), (0x46546c67, 2, len(data)))
        size, kind = struct.unpack_from('<II', data, 12)
        self.assertEqual(kind, 0x4e4f534a)
        metadata = json.loads(data[20:20 + size])
        primitive = metadata['meshes'][0]['primitives'][0]
        self.assertEqual(metadata['accessors'][primitive['indices']]['count'] // 3, 96781)
        self.assertEqual(len(metadata['images']), 2)

    def test_png_text_metadata_removed(self):
        for path in (ROOT / 'examples').glob('*.png'):
            data = path.read_bytes()
            offset = 8
            while offset < len(data):
                length = struct.unpack_from('>I', data, offset)[0]
                self.assertNotIn(data[offset + 4:offset + 8], (b'tEXt', b'iTXt', b'zTXt'))
                offset += length + 12
            self.assertEqual(offset, len(data))

    def test_failed_workflow_is_reported_and_history_retained(self):
        submit = script('submit_workflow')
        graph = json.loads((ROOT / 'profiles/z-image/workflow-api.json').read_text())
        info = {node['class_type']: {} for node in graph.values()}
        error = {'status': {'status_str': 'error'}, 'outputs': {}}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'request'
            arguments = ['submit_workflow.py', '--workflow', str(ROOT / 'profiles/z-image/workflow-api.json'), '--out-dir', str(output)]
            with patch.object(sys, 'argv', arguments), patch.object(submit, 'request', side_effect=[info, {'prompt_id': 'test'}, {'test': error}]):
                with self.assertRaisesRegex(RuntimeError, 'Workflow failed'):
                    submit.main()
            self.assertEqual(json.loads((output / 'history.json').read_text()), error)

    def test_cancel_never_uses_global_interrupt(self):
        submit = script('submit_workflow')
        running = {'queue_running': [[0, 'mine']], 'queue_pending': []}
        with patch.object(submit, 'request', return_value=running) as request:
            submit.cancel_own_prompt('http://127.0.0.1:8188', 'mine')
            self.assertEqual(request.call_count, 1)
        pending = {'queue_running': [[0, 'someone-else']], 'queue_pending': [[1, 'mine']]}
        with patch.object(submit, 'request', side_effect=[pending, {}]) as request:
            submit.cancel_own_prompt('http://127.0.0.1:8188', 'mine')
            self.assertEqual(request.call_args.kwargs, {'value': {'delete': ['mine']}})


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Install Torch to run the CPU numerical checks')
class MathTests(unittest.TestCase):
    def test_attention_masks_gqa_and_sparse_math_against_cpu_reference(self):
        from mi50.ops import verify_sdpa, verify_sparse_math
        torch.set_num_threads(2)
        attention = verify_sdpa('cpu')
        self.assertEqual(len(attention), 4)
        self.assertLess(max(value['max_error'] for value in attention), .000002)
        sparse = verify_sparse_math('cpu')
        self.assertLess(sparse['sparse_conv_error'], .00001)
        self.assertLess(sparse['weighted_gather_error'], .00001)

    def test_empty_attention_and_missing_neighbors(self):
        from mi50.ops import bounded_sdpa, weighted_gather
        value = bounded_sdpa(torch.empty(1, 2, 0, 4), torch.zeros(1, 2, 3, 4), torch.zeros(1, 2, 3, 6))
        self.assertEqual(tuple(value.shape), (1, 2, 0, 6))
        features = torch.ones(2, 3)
        indices = torch.tensor([[-1, 0xffffffff]])
        result = weighted_gather(features, indices, torch.ones(1, 2))
        self.assertTrue(torch.equal(result, torch.zeros(1, 3)))
        with self.assertRaisesRegex(ValueError, 'dropout'):
            bounded_sdpa(torch.ones(1, 1, 4), torch.ones(1, 1, 4), torch.ones(1, 1, 4), dropout_p=.1)


@unittest.skipUnless(sys.platform == 'win32', 'Windows Job test')
class JobTests(unittest.TestCase):
    def test_closing_job_stops_own_child_and_leaves_other_process_alive(self):
        from mi50.windows_job import ChildJob
        other = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(90)'])
        try:
            with ChildJob() as job:
                child = job.start([sys.executable, '-c', 'import time;time.sleep(90)'])
                self.assertIsNone(child.poll())
            child.wait(timeout=10)
            self.assertIsNone(other.poll())
        finally:
            other.terminate()
            other.wait(timeout=10)


if __name__ == '__main__':
    unittest.main()
