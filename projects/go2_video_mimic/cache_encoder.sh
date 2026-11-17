#!/usr/bin/env bash
# Reuse a locally installed static encoder; never downloads an image/package.
set -euo pipefail
project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
target="$project/.cache/ffmpeg"
[[ ! -x "$target" ]] || exit 0
mkdir -p "$project/.cache"
if [[ -n "${VIDEO_FFMPEG:-}" ]]; then
    cp -- "$VIDEO_FFMPEG" "$target.tmp"
else
    source_image="${VIDEO_FFMPEG_SOURCE_IMAGE:-dog2go2:cu121}"
    docker image inspect "$source_image" >/dev/null 2>&1 || {
        echo 'Video encoder unavailable. Set VIDEO_FFMPEG to a Linux static ffmpeg binary.' >&2
        exit 1
    }
    docker run --rm --pull=never --network none --entrypoint /usr/local/bin/python \
        "$source_image" -c 'import sys,imageio_ffmpeg;sys.stdout.buffer.write(open(imageio_ffmpeg.get_ffmpeg_exe(),"rb").read())' > "$target.tmp"
    docker image inspect "$source_image" --format '{{.Id}}' > "$project/.cache/ffmpeg_source_image.txt"
fi
chmod 755 "$target.tmp"
mv -- "$target.tmp" "$target"
