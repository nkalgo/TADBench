"""A simple web server to display the HTML page rendered by a function."""
import sys
import threading
import traceback
from http.server import SimpleHTTPRequestHandler, HTTPServer

import click

__all__ = ['serve_html']


def serve_html(render_fn, host='127.0.0.1', port=0, open_browser: bool = True):
    class ServerHandler(SimpleHTTPRequestHandler):
        def do_GET(self):
            try:
                cnt = render_fn()
                if isinstance(cnt, str):
                    cnt = cnt.encode('utf-8')
                code = 200
                mime_type = 'text/html'
            except Exception:
                cnt = traceback.format_exc().encode('utf-8')
                code = 500
                mime_type = 'text/plain'

            self.protocol_version = 'HTTP/1.1'
            self.send_response(code)
            self.send_header('Content-type', mime_type)
            self.end_headers()
            self.wfile.write(cnt)

    def thread_func():
        server.serve_forever()

    server = HTTPServer((host, port), ServerHandler)
    sck = server.socket
    laddr = sck.getsockname()
    url = f'http://{laddr[0]}:{laddr[1]}'
    print(f'Server started at {url}', file=sys.stderr)

    thread = threading.Thread(target=thread_func)
    thread.start()

    if open_browser:
        try:
            click.launch(url)
        except Exception as ex:
            print(f'{traceback.format_exc()}\nFailed to open web browser.', file=sys.stderr)

    try:
        thread.join()
    finally:
        server.shutdown()
        thread.join()
