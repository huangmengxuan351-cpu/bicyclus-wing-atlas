"""Check the published display after CI deployment; never claim prediction readiness."""
import argparse
import json
import subprocess


def request(url):
    response = subprocess.run(['curl', '--silent', '--show-error', '--location',
                               '--max-time', '60', '--retry', '2', '--write-out', '\n%{http_code}',
                               url], capture_output=True, check=True)
    body, code = response.stdout.rsplit(b'\n', 1)
    return int(code), body


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', choices=('zh', 'en'), required=True)
    args = parser.parse_args()
    origin = 'https://bicyclus-wing-atlas-' + args.language + '.pages.dev'
    checks = []
    for path in ('/', '/__deployment_health', '/' + args.language + '/atlas/',
                 '/' + args.language + '/anatomy/', '/' + args.language + '/predict/'):
        code, body = request(origin + path)
        checks.append({'path': path, 'passed': code == 200, 'status': code})
    for path in ('/downloads/', '/atlas-assets/probe.png'):
        code, _ = request(origin + path)
        checks.append({'path': path, 'passed': code == 401, 'status': code})
    print(json.dumps({'language': args.language, 'checks': checks,
                      'prediction_readiness_tested': False}, indent=2))
    if not all(check['passed'] for check in checks):
        raise RuntimeError('Published display or anonymous original-download guard failed.')


if __name__ == '__main__':
    main()
