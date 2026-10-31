#!/usr/bin/env bash
set -euo pipefail
[[ $(id -u) == 0 ]]
[[ $# == 1 ]] || { echo 'Usage: enable-https.sh <expected VM public IPv4>'; exit 1; }
python3 - "$1" <<'PY'
import socket,sys
expected = sys.argv[1]
addresses = {x[4][0] for x in socket.getaddrinfo('voices.benchmarks.cloud', 80, proto=socket.IPPROTO_TCP)}
if addresses != {expected}:
    raise SystemExit('DNS must resolve only to the expected VM before certificate issuance.')
PY
certbot certonly --webroot -w /var/www/ampixa-arena-acme \
  --cert-name voices.benchmarks.cloud -d voices.benchmarks.cloud --non-interactive
config=/etc/nginx/sites-available/ampixa-arena
backup=$(mktemp)
cp "$config" "$backup"
cat > "$config" <<'CONF'
server {
    listen 80;
    server_name voices.benchmarks.cloud;
    location /.well-known/acme-challenge/ { root /var/www/ampixa-arena-acme; }
    location / { return 301 https://voices.benchmarks.cloud$request_uri; }
}
server {
    listen 443 ssl;
    server_name voices.benchmarks.cloud;
    ssl_certificate /etc/letsencrypt/live/voices.benchmarks.cloud/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/voices.benchmarks.cloud/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    location / {
        client_max_body_size 150m;
        proxy_pass http://127.0.0.1:5193;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $remote_addr;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_request_buffering off;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }
}
CONF
if ! nginx -t; then
    cp "$backup" "$config"
    rm "$backup"
    exit 1
fi
systemctl reload nginx
rm "$backup"
install -d /etc/letsencrypt/renewal-hooks/deploy
cat > /etc/letsencrypt/renewal-hooks/deploy/ampixa-arena-reload <<'HOOK'
#!/bin/sh
/usr/sbin/nginx -t && /usr/bin/systemctl reload nginx
HOOK
chmod 755 /etc/letsencrypt/renewal-hooks/deploy/ampixa-arena-reload
curl -fsS https://voices.benchmarks.cloud/api/benchmark >/dev/null
echo 'Arena HTTPS verified.'
