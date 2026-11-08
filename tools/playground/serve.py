"""Serve the optional Mora-3 playground and a local checkpoint adapter on loopback."""
from __future__ import annotations

import argparse
import base64
import binascii
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import import_module
import json
from mimetypes import guess_type
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit

PROTOCOL = 'mora3-playground-v1'
MIB = 1024 * 1024
REQUEST_LIMIT = 17 * MIB
RESULT_LIMIT = MIB
TASKS = {
    'affinity': ('predict',), 'annotation': ('predict',), 'expression': ('predict',),
    'bioelectric': ('predict',), 'answer': ('predict',), 'structure': ('generate',),
    'dynamics': ('generate',), 'sequence': ('generate',), 'molecule': ('generate',),
    'image': ('predict', 'generate'), 'reaction': ('generate',),
}
MODALITIES = {'sequence', 'molecule', 'structure', 'dynamics', 'expression', 'image', 'assay', 'reaction', 'text'}
BUNDLE = Path(__file__).resolve().parent
PUBLIC_SITE = BUNDLE.parents[1] / 'docs' if BUNDLE.parent.name == 'tools' else BUNDLE.parent / 'dist'


class RequestError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def checked_string(value, limit):
    return isinstance(value, str) and 0 < len(value) <= limit


def validate_request(value):
    def fail(message):
        raise RequestError(HTTPStatus.BAD_REQUEST, message)

    if not isinstance(value, dict) or set(value) != {'protocol', 'task', 'output', 'messages', 'inputs'}:
        fail('The request does not match the local model contract.')
    if value['protocol'] != PROTOCOL or not isinstance(value['output'], str) or value['task'] not in TASKS.get(value['output'], ()):
        fail('Choose a valid output and matching model task.')
    messages, inputs = value['messages'], value['inputs']
    if not isinstance(messages, list) or len(messages) > 20 or not isinstance(inputs, list) or len(inputs) > 8:
        fail('Requests allow up to 20 turns and eight attachments.')
    for entry in messages:
        if (not isinstance(entry, dict) or set(entry) != {'role', 'content'} or
                entry['role'] not in ('user', 'assistant') or not checked_string(entry['content'], 8000)):
            fail('A conversation turn is invalid or too long.')
    if not inputs and not any(entry['role'] == 'user' for entry in messages):
        fail('Add an instruction or an attachment.')
    total = 0
    for entry in inputs:
        if not isinstance(entry, dict) or set(entry) != {'modality', 'name', 'bytes', 'encoding', 'data'}:
            fail('An attachment is missing required metadata.')
        name, modality, size = entry['name'], entry['modality'], entry['bytes']
        if (not isinstance(modality, str) or modality not in MODALITIES or not checked_string(name, 255) or
                any(ch in name for ch in ('/', '\\', '\x00', '\n', '\r')) or name in ('.', '..') or
                type(size) is not int or not 1 <= size <= 6 * MIB or entry['encoding'] != 'base64'):
            fail('An attachment has an invalid name, modality, or size.')
        total += size
        if total > 12 * MIB or not checked_string(entry['data'], 4 * ((size + 2) // 3)):
            fail('Attachment data exceeds its declared size or the total file limit.')
        try:
            decoded_size = len(base64.b64decode(entry['data'], validate=True))
        except (binascii.Error, ValueError):
            fail('An attachment is not valid base64.')
        if decoded_size != size:
            fail('An attachment does not match its declared byte count.')
    return value


def validate_capabilities(value):
    if (not isinstance(value, dict) or not checked_string(value.get('model'), 160) or
            not checked_string(value.get('checkpoint'), 240) or not isinstance(value.get('outputs'), list) or
            not 1 <= len(value['outputs']) <= len(TASKS)):
        raise ValueError('The adapter must declare a model, checkpoint, and supported outputs.')
    outputs = []
    for item in value['outputs']:
        if not isinstance(item, dict) or not isinstance(item.get('id'), str):
            raise ValueError('Invalid output declaration.')
        output = item['id']
        tasks = item.get('tasks')
        if (output not in TASKS or not isinstance(tasks, list) or not tasks or
                any(task not in TASKS[output] for task in tasks) or len(set(tasks)) != len(tasks) or
                any(row['id'] == output for row in outputs)):
            raise ValueError('An output declaration is unsupported or repeated.')
        declared = {'id': output, 'tasks': item['tasks']}
        if checked_string(item.get('notes'), 500):
            declared['notes'] = item['notes']
        outputs.append(declared)
    return {'model': value['model'], 'checkpoint': value['checkpoint'], 'outputs': outputs}


def checkpoints(weights_dir, weights_root):
    if weights_dir is None and weights_root is None:
        return {}
    configured = (weights_dir if weights_dir is not None else weights_root).resolve(strict=True)
    if not configured.is_dir():
        raise ValueError('The configured weights location must be a directory.')
    directories = [configured] if weights_dir is not None else sorted(
        (path for path in configured.iterdir() if not path.is_symlink() and path.is_dir() and not path.name.startswith('.')),
        key=lambda path: path.name.casefold(),
    )
    if len(directories) > 128:
        raise ValueError('Choose a weights root with at most 128 immediate checkpoint directories.')
    return {f'c{index + 1}': path for index, path in enumerate(directories)}


class Runtime:
    def __init__(self, candidates, factory):
        self.candidates = candidates
        self.factory = factory
        self.lock = Lock()
        self.active_id = None
        self.adapter = None
        self.capabilities = None
        self.revision = 0

    def _status(self):
        if self.adapter:
            state, detail = 'ready', 'The local checkpoint is loaded. Only the listed model outputs are enabled.'
        elif self.candidates and self.factory:
            state, detail = 'select', 'Choose a checkpoint directory and load it to check supported outputs.'
        elif self.candidates:
            state, detail = 'unavailable', 'Checkpoint directories are visible. Relaunch with a compatible local adapter to load one.'
        else:
            state, detail = 'preview', 'No local checkpoint directory is configured. Synthetic examples are available.'
        return {
            'protocol': PROTOCOL, 'state': state, 'detail': detail,
            'can_select': bool(self.factory and self.candidates),
            'checkpoints': [{'id': id, 'label': path.name} for id, path in self.candidates.items()],
            'active': ({'id': self.active_id, 'revision': self.revision, **self.capabilities} if self.adapter else None),
        }

    def status(self):
        with self.lock:
            return self._status()

    def select(self, selection):
        if not isinstance(selection, dict) or set(selection) != {'id'} or not isinstance(selection['id'], str):
            raise RequestError(HTTPStatus.BAD_REQUEST, 'Select a checkpoint from the listed directories.')
        id = selection['id']
        if id not in self.candidates:
            raise RequestError(HTTPStatus.NOT_FOUND, 'This checkpoint directory is not in the configured list.')
        if not self.factory:
            raise RequestError(HTTPStatus.SERVICE_UNAVAILABLE, 'The local model adapter is unavailable.')
        with self.lock:
            path = self.candidates[id]
            try:
                if path.is_symlink() or not path.is_dir() or path.resolve(strict=True) != path:
                    raise ValueError('Checkpoint directory changed.')
                adapter = self.factory(path)
                if not callable(getattr(adapter, 'infer', None)):
                    raise ValueError('Missing inference method.')
                capabilities = validate_capabilities(adapter.capabilities())
            except Exception:
                raise RequestError(HTTPStatus.UNPROCESSABLE_ENTITY,
                                   'This directory could not be loaded. Check the checkpoint manifest, files, and adapter version.') from None
            self.adapter = adapter
            self.capabilities = capabilities
            self.active_id = id
            self.revision += 1
            return self._status()

    def infer(self, payload, revision):
        request = validate_request(payload)
        with self.lock:
            if not self.adapter or revision != str(self.revision):
                raise RequestError(HTTPStatus.CONFLICT, 'The active checkpoint changed. Refresh its status and select the checkpoint again.')
            if not any(row['id'] == request['output'] and request['task'] in row['tasks'] for row in self.capabilities['outputs']):
                raise RequestError(HTTPStatus.UNPROCESSABLE_ENTITY, 'This checkpoint does not support the requested task and output.')
            try:
                result = self.adapter.infer(request)
            except (ValueError, NotImplementedError):
                raise RequestError(HTTPStatus.UNPROCESSABLE_ENTITY,
                                   'The checkpoint could not process this request. Check the supported input notes and attached files.') from None
            except Exception:
                raise RequestError(HTTPStatus.INTERNAL_SERVER_ERROR, 'The local model did not complete the request.') from None
            if (not isinstance(result, dict) or result.get('protocol') != PROTOCOL or
                    result.get('output') != request['output'] or not checked_string(result.get('provenance'), 1000)):
                raise RequestError(HTTPStatus.BAD_GATEWAY, 'The local adapter returned an invalid or unbound result.')
            try:
                data = json.dumps(result, ensure_ascii=False, allow_nan=False).encode('utf-8')
            except (TypeError, ValueError, UnicodeError):
                raise RequestError(HTTPStatus.BAD_GATEWAY, 'The result cannot be represented as finite JSON.') from None
            if len(data) > RESULT_LIMIT:
                raise RequestError(HTTPStatus.BAD_GATEWAY, 'The local model result exceeds the browser display limit.')
            return data


class LocalServer(ThreadingHTTPServer):
    def __init__(self, port, directory, public, runtime):
        super().__init__(('127.0.0.1', port), LocalHandler)
        self.directory = directory.resolve(strict=True)
        self.public = public.resolve(strict=True)
        self.runtime = runtime
        self.host = f'127.0.0.1:{self.server_port}'
        self.origin = f'http://{self.host}'


class LocalHandler(BaseHTTPRequestHandler):
    server_version = 'Mora3Local/1'

    def log_message(self, *_):
        pass

    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def send_data(self, status, data, media_type):
        self.send_response(status)
        self.send_header('Content-Type', media_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Cross-Origin-Resource-Policy', 'same-origin')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('Content-Security-Policy',
                         "default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
                         "img-src 'self' data:; connect-src 'self'; font-src 'self'; base-uri 'none'; "
                         "frame-ancestors 'none'; form-action 'self'")
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(data)

    def send_json(self, status, data):
        self.send_data(status, json.dumps(data, ensure_ascii=False).encode('utf-8'), 'application/json; charset=utf-8')

    def authorize(self, *, write=False):
        if self.headers.get_all('Host') != [self.server.host]:
            raise RequestError(HTTPStatus.FORBIDDEN, 'Use the loopback URL printed by the launcher.')
        origin = self.headers.get_all('Origin')
        if (write and origin != [self.server.origin]) or (origin is not None and origin != [self.server.origin]):
            raise RequestError(HTTPStatus.FORBIDDEN, 'The local service accepts requests from its own origin only.')
        if self.headers.get('Sec-Fetch-Site') not in (None, 'same-origin', 'none'):
            raise RequestError(HTTPStatus.FORBIDDEN, 'Cross-origin requests are disabled.')

    def request_json(self, limit):
        if (self.headers.get_content_type() != 'application/json' or self.headers.get('Transfer-Encoding') or
                len(self.headers.get_all('Content-Length', [])) != 1):
            raise RequestError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, 'Send a single-length JSON request.')
        try:
            length = int(self.headers['Content-Length'])
        except (TypeError, ValueError):
            raise RequestError(HTTPStatus.BAD_REQUEST, 'The request length is invalid.') from None
        if not 0 < length <= limit:
            raise RequestError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, 'The request exceeds the local upload limit.')
        try:
            return json.loads(self.rfile.read(length).decode('utf-8'))
        except (ValueError, UnicodeError, RecursionError):
            raise RequestError(HTTPStatus.BAD_REQUEST, 'Send valid UTF-8 JSON.') from None

    def do_GET(self):
        try:
            self.authorize()
            path = urlsplit(self.path).path
            if path == '/api/runtime':
                self.send_json(HTTPStatus.OK, self.server.runtime.status())
                return
            if path == '/':
                path = '/playground.html'
            segments = path.strip('/').split('/')
            supported = (len(segments) == 1 and path.endswith('.html')) or (len(segments) == 2 and segments[0] == 'assets')
            if not supported or any(not part or part.startswith('.') or '%' in part for part in segments):
                raise RequestError(HTTPStatus.NOT_FOUND, 'This path is not part of the local playground.')
            file = self.server.directory.joinpath(*segments)
            if not file.is_file():
                file = self.server.public.joinpath(*segments)
            if not file.is_file() or file.resolve() != file:
                raise RequestError(HTTPStatus.NOT_FOUND, 'This path is not part of the local playground.')
            media_type = guess_type(file.name)[0] or 'application/octet-stream'
            self.send_data(HTTPStatus.OK, file.read_bytes(), media_type)
        except RequestError as error:
            self.send_json(error.status, {'error': str(error)})

    do_HEAD = do_GET

    def do_POST(self):
        try:
            self.authorize(write=True)
            if self.path == '/api/checkpoint':
                self.send_json(HTTPStatus.OK, self.server.runtime.select(self.request_json(1024)))
            elif self.path == '/api/infer':
                result = self.server.runtime.infer(self.request_json(REQUEST_LIMIT), self.headers.get('X-Mora3-Revision'))
                self.send_data(HTTPStatus.OK, result, 'application/json; charset=utf-8')
            else:
                raise RequestError(HTTPStatus.NOT_FOUND, 'This path is not part of the local model service.')
        except RequestError as error:
            self.send_json(error.status, {'error': str(error)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--weights-dir', type=Path, help='One local checkpoint directory.')
    group.add_argument('--weights-root', type=Path, help='A directory containing immediate checkpoint subdirectories.')
    parser.add_argument('--adapter', help='Trusted local module:factory, imported from the selected Python environment.')
    parser.add_argument('--port', type=int, default=8632)
    args = parser.parse_args()
    if not (BUNDLE / 'playground.html').is_file() or not (PUBLIC_SITE / 'assets' / 'site.css').is_file():
        parser.error('Keep the local tool bundle and the public site assets in their original repository layout.')
    try:
        candidates = checkpoints(args.weights_dir, args.weights_root)
    except (OSError, ValueError):
        parser.error('Choose an existing local weights directory or a root with at most 128 checkpoint directories.')
    factory = None
    if args.adapter:
        module, separator, name = args.adapter.partition(':')
        if not separator or not name.isidentifier() or not all(part.isidentifier() for part in module.split('.')):
            parser.error('Use a trusted adapter name in module:factory format.')
        try:
            factory = getattr(import_module(module), name)
        except (ImportError, AttributeError):
            parser.error('The local adapter is unavailable in this Python environment.')
        if not callable(factory):
            parser.error('The selected local adapter factory must be callable.')
    if not 0 <= args.port <= 65535:
        parser.error('Choose a valid TCP port.')
    with LocalServer(args.port, BUNDLE, PUBLIC_SITE, Runtime(candidates, factory)) as server:
        print(f'Local Mora-3 playground: {server.origin}/playground.html', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
