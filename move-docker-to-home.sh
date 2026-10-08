#!/usr/bin/env bash
set -Eeuo pipefail

# One-time migration for this host's Docker Engine and external containerd store.
target=/home/docker-storage
docker_config=/etc/docker/daemon.json
containerd_config=/etc/containerd/config.toml
docker_dropin=/etc/systemd/system/docker.service.d/10-home-storage.conf
containerd_dropin=/etc/systemd/system/containerd.service.d/10-home-storage.conf

if (( EUID != 0 )); then
    echo "Run this script with sudo." >&2
    exit 1
fi
if [[ "$(findmnt -n -o TARGET --target /home)" != /home ]]; then
    echo "/home is not mounted as its own filesystem; refusing to migrate." >&2
    exit 1
fi
if [[ -e "$target" || -e "$docker_config" || -e "$docker_dropin" || -e "$containerd_dropin" ]]; then
    echo "Destination or configuration already exists; refusing to overwrite it." >&2
    exit 1
fi
if ! grep -Fxq '#root = "/var/lib/containerd"' "$containerd_config"; then
    echo "containerd configuration differs from the inspected version; refusing to edit it." >&2
    exit 1
fi
if [[ -n "$(docker ps -aq)" ]]; then
    echo "Docker has containers. Stop and review them before migration." >&2
    exit 1
fi

image_list=$(mktemp)
backup_config=$(mktemp /etc/containerd/config.toml.before-home.XXXXXX)
docker image ls --format '{{.Repository}}:{{.Tag}} {{.ID}}' | sort > "$image_list"
cp -a "$containerd_config" "$backup_config"

rollback() {
    local status=$?
    trap - ERR
    echo "Migration failed (exit $status). Restoring configuration and starting old storage." >&2
    cp -a "$backup_config" "$containerd_config"
    rm -f "$docker_config" "$docker_dropin" "$containerd_dropin"
    systemctl daemon-reload || true
    systemctl start containerd.service docker.socket docker.service || true
    rm -f "$image_list"
    exit "$status"
}
trap rollback ERR

echo "Stopping Docker and containerd..."
systemctl stop docker.socket docker.service containerd.service

install -d -m 0700 -o root -g root "$target" "$target/docker" "$target/containerd"
echo "Copying existing Docker data to $target..."
rsync -aHAX --numeric-ids /var/lib/docker/ "$target/docker/"
rsync -aHAX --numeric-ids /var/lib/containerd/ "$target/containerd/"

printf '{"data-root": "%s/docker"}\n' "$target" > "$docker_config"
sed -i 's@^#root = "/var/lib/containerd"$@root = "/home/docker-storage/containerd"@' "$containerd_config"
install -d -m 0755 /etc/systemd/system/docker.service.d /etc/systemd/system/containerd.service.d
printf '[Unit]\nRequiresMountsFor=%s\n' "$target" > "$docker_dropin"
printf '[Unit]\nRequiresMountsFor=%s\n' "$target" > "$containerd_dropin"

systemctl daemon-reload
echo "Starting Docker with storage under /home..."
systemctl start containerd.service docker.socket docker.service
test "$(docker info --format '{{.DockerRootDir}}')" = "$target/docker"
test "$(findmnt -n -o TARGET --target "$target/containerd")" = /home
docker image ls --format '{{.Repository}}:{{.Tag}} {{.ID}}' | sort | cmp - "$image_list"

trap - ERR
rm -f "$image_list"
echo "Migration verified. Original /var/lib/docker and /var/lib/containerd remain for rollback."
docker info --format 'Docker Root Dir: {{.DockerRootDir}}'
docker image ls --format '{{.Repository}}:{{.Tag}} {{.Size}}'
du -sh "$target/docker" "$target/containerd"
df -h / /home
