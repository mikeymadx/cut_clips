#!/usr/bin/env python3
"""Planning JSON editor — local web UI for reviewing and annotating plan_video.py output.

Usage:
    python tools/edit_planning.py --json path/to/scenes_planning.json
    python tools/edit_planning.py --json path/to/scenes_planning.json --video path/to/debug.mp4 --port 5050
"""

import argparse
import json
import os
import threading
import webbrowser
from pathlib import Path

from flask import Flask, Response, jsonify, request

app = Flask(__name__)
STATE = {}

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Plan Editor</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
           font-size: 13px; background: #111; color: #ddd; height: 100vh;
           display: flex; flex-direction: column; overflow: hidden; }

    /* Header */
    #header { display: flex; align-items: center; gap: 12px; padding: 8px 14px;
               background: #1a1a1a; border-bottom: 1px solid #333; flex-shrink: 0; }
    #filename { font-weight: 600; color: #aaa; font-size: 12px; flex: 1;
                overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    #save-btn { padding: 5px 16px; border: none; border-radius: 4px;
                background: #2a6496; color: #fff; cursor: pointer; font-size: 12px;
                font-weight: 600; transition: background 0.15s; }
    #save-btn:hover { background: #3a74a6; }
    #save-btn.dirty { background: #c87e00; }
    #save-btn.dirty:hover { background: #d98e10; }
    #status { font-size: 11px; color: #888; min-width: 120px; }
    #status.ok { color: #5a9; }
    #status.err { color: #e55; }

    /* Middle: video + timeline */
    #middle { display: flex; flex: 0 0 auto; height: 42vh; border-bottom: 1px solid #333; }

    #video-panel { flex: 0 0 58%; display: flex; align-items: center;
                   justify-content: center; background: #000; overflow: hidden; }
    #player { max-width: 100%; max-height: 100%; display: block; }
    #no-video { color: #555; font-size: 14px; }

    #timeline-panel { flex: 1; overflow-y: auto; padding: 6px 8px;
                      background: #161616; }
    #timeline-panel h4 { font-size: 11px; color: #555; margin-bottom: 6px;
                         text-transform: uppercase; letter-spacing: 0.05em; }
    .t-seg { border-radius: 3px; margin-bottom: 3px; padding: 3px 7px;
              cursor: pointer; border-left: 3px solid transparent;
              transition: filter 0.1s; font-size: 11px; line-height: 1.4;
              overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
    .t-seg:hover { filter: brightness(1.2); }
    .t-seg.active { border-left-color: #fff; filter: brightness(1.35); }
    .t-seg.selected { outline: 1px solid #fff; }
    .t-face { background: #1d3a5a; color: #7ab8e8; }
    .t-body { background: #3a2a00; color: #e8b847; }
    .t-low  { background: #3a1010; color: #e88080; }

    /* Table */
    #table-wrap { flex: 1; overflow: auto; }
    table { width: 100%; border-collapse: collapse; }
    thead th { position: sticky; top: 0; background: #1e1e1e; padding: 6px 8px;
               text-align: left; font-size: 11px; color: #777;
               border-bottom: 1px solid #333; white-space: nowrap; }
    tbody tr { border-bottom: 1px solid #222; cursor: pointer; }
    tbody tr:hover { background: #1a1a1a; }
    tbody tr.selected { background: #1d2d3d; }
    tbody tr.active-row { background: #1a2a1a; }
    td { padding: 4px 8px; vertical-align: middle; }
    td.ro { color: #666; font-size: 11px; }
    input[type=number], input[type=text] {
      background: #222; border: 1px solid #444; color: #ddd;
      padding: 2px 5px; border-radius: 3px; font-size: 12px; width: 100%; }
    input[type=number] { width: 80px; }
    input[type=text] { min-width: 90px; }
    input.invalid { border-color: #e55; background: #2a1111; }
    input:focus { outline: 1px solid #4a8abf; border-color: #4a8abf; }

    .detail-row { display: none; }
    .detail-row.open { display: table-row; }
    .detail-row td { background: #0d1a0d; padding: 6px 16px 8px 32px; }
    .detail-table { border-collapse: collapse; font-size: 11px; }
    .detail-table th { color: #555; padding: 2px 10px 2px 0; text-align: left; font-weight: normal; }
    .detail-table td { color: #aaa; padding: 2px 10px 2px 0; }
    .detail-table tr.focus-track td { color: #5a9; }
    .detail-no-tracks { color: #555; font-size: 11px; font-style: italic; }
    .split-btn { background: #2a2a2a; border: 1px solid #444; color: #777; cursor: pointer;
                 border-radius: 3px; padding: 1px 6px; font-size: 13px; }
    .split-btn:hover { background: #383838; color: #ddd; }
    .split-ok-btn { background: #1a5c2a; border: none; color: #aee; cursor: pointer;
                    border-radius: 3px; padding: 1px 5px; font-size: 12px; margin: 0 2px; }
    .split-ok-btn:hover { background: #226e34; }
    .split-cancel-btn { background: #4a1a1a; border: none; color: #eaa; cursor: pointer;
                        border-radius: 3px; padding: 1px 5px; font-size: 12px; }
    .split-cancel-btn:hover { background: #5e2222; }
  </style>
</head>
<body>
  <div id="header">
    <span id="filename"></span>
    <button id="save-btn" onclick="saveData()">Save</button>
    <span id="status"></span>
  </div>
  <div id="middle">
    <div id="video-panel">
      <video id="player" controls></video>
      <div id="no-video" style="display:none">No video provided — pass --video to enable</div>
    </div>
    <div id="timeline-panel">
      <h4>Segments</h4>
      <div id="timeline"></div>
    </div>
  </div>
  <div id="table-wrap">
    <table>
      <thead>
        <tr>
          <th>#</th><th>start_s</th><th>end_s</th><th>dur_s</th>
          <th>trigger</th><th>mode</th><th>det%</th><th>avg_x</th><th>std_x</th>
          <th>tracks</th><th>focus_id</th><th>importance</th><th>face</th><th>area%</th>
          <th>label</th><th>subject_focus</th><th>description</th><th>pan_plan</th>
          <th></th>
        </tr>
      </thead>
      <tbody id="tbody"></tbody>
    </table>
  </div>

  <script>
    let segments = [];
    let dirty = false;
    let selectedIdx = -1;
    const video = document.getElementById('player');

    function setDirty(val) {
      dirty = val;
      const btn = document.getElementById('save-btn');
      btn.className = val ? 'dirty' : '';
    }

    function setStatus(msg, cls) {
      const el = document.getElementById('status');
      el.textContent = msg;
      el.className = cls || '';
    }

    function segColor(seg) {
      if (seg.detection_rate < 0.7) return 't-low';
      return seg.dominant_mode === 'face' ? 't-face' : 't-body';
    }

    function renderTimeline() {
      const total = segments.reduce((s, x) => s + x.duration_s, 0) || 1;
      const MIN_H = 18, container = document.getElementById('timeline');
      const panelH = document.getElementById('timeline-panel').clientHeight - 28;
      container.innerHTML = '';
      segments.forEach((seg, i) => {
        const h = Math.max(MIN_H, (seg.duration_s / total) * panelH);
        const div = document.createElement('div');
        div.className = 't-seg ' + segColor(seg);
        div.id = 'tseg-' + i;
        div.style.height = h + 'px';
        const lbl = seg.label || seg.trigger || '';
        div.title = `Seg ${seg.segment} | ${seg.dominant_mode} | ${seg.start_s}s–${seg.end_s}s`;
        div.textContent = `${seg.segment}  ${lbl}  ${seg.duration_s.toFixed(1)}s`;
        div.addEventListener('click', () => selectSegment(i, true));
        container.appendChild(div);
      });
    }

    function refreshRow(idx) {
      const tr = document.getElementById('trow-' + idx);
      if (!tr) return;
      const seg = segments[idx];
      const nums = tr.querySelectorAll('input[type=number]');
      if (nums[0]) nums[0].value = seg.start_s;
      if (nums[1]) nums[1].value = seg.end_s;
      const durCell = tr.querySelector('.dur-cell');
      if (durCell) durCell.textContent = seg.duration_s.toFixed(2);
    }

    function renderTable() {
      const tbody = document.getElementById('tbody');
      tbody.innerHTML = '';
      segments.forEach((seg, i) => {
        const tr = document.createElement('tr');
        tr.className = 'trow';
        tr.id = 'trow-' + i;
        tr.addEventListener('click', () => selectSegment(i, true));

        function ro(val) {
          const td = document.createElement('td');
          td.className = 'ro';
          td.textContent = val ?? '';
          return td;
        }
        function importanceTd(val) {
          const td = document.createElement('td');
          td.className = 'ro';
          if (val == null || val < 0) { td.textContent = '—'; return td; }
          td.textContent = val.toFixed(2);
          if (val >= 0.7) td.style.color = '#5a9';
          else if (val >= 0.4) td.style.color = '#c87e00';
          else td.style.color = '#e55';
          return td;
        }
        function numInput(field, val) {
          const td = document.createElement('td');
          const inp = document.createElement('input');
          inp.type = 'number'; inp.step = '0.01'; inp.value = val;
          inp.addEventListener('click', e => e.stopPropagation());
          inp.addEventListener('input', () => {
            const v = parseFloat(inp.value);
            if (isNaN(v)) return;
            segments[i][field] = v;
            // cascade to adjacent segment so no gaps or overlaps
            if (field === 'end_s' && i + 1 < segments.length) {
              segments[i + 1].start_s = v;
              segments[i + 1].duration_s = +(segments[i + 1].end_s - v).toFixed(3);
              refreshRow(i + 1);
            } else if (field === 'start_s' && i > 0) {
              segments[i - 1].end_s = v;
              segments[i - 1].duration_s = +(v - segments[i - 1].start_s).toFixed(3);
              refreshRow(i - 1);
            }
            // recalc own duration
            const dur = +(segments[i].end_s - segments[i].start_s).toFixed(3);
            segments[i].duration_s = dur;
            const durCell = tr.querySelector('.dur-cell');
            if (durCell) durCell.textContent = dur.toFixed(2);
            inp.classList.toggle('invalid', segments[i].start_s >= segments[i].end_s);
            setDirty(true);
            renderTimeline();
          });
          td.appendChild(inp); return td;
        }
        function txtInput(field, val) {
          const td = document.createElement('td');
          const inp = document.createElement('input');
          inp.type = 'text'; inp.value = val || '';
          inp.addEventListener('click', e => e.stopPropagation());
          inp.addEventListener('input', () => {
            segments[i][field] = inp.value;
            setDirty(true);
            renderTimeline();
          });
          td.appendChild(inp); return td;
        }

        const durTd = ro(seg.duration_s.toFixed(2));
        durTd.className = 'ro dur-cell';

        tr.appendChild(ro(seg.segment));
        tr.appendChild(numInput('start_s', seg.start_s));
        tr.appendChild(numInput('end_s', seg.end_s));
        tr.appendChild(durTd);
        tr.appendChild(ro(seg.trigger));
        tr.appendChild(ro(seg.dominant_mode));
        tr.appendChild(ro((seg.detection_rate * 100).toFixed(0) + '%'));
        tr.appendChild(ro(seg.avg_crop_x?.toFixed(0)));
        tr.appendChild(ro(seg.crop_x_std?.toFixed(0)));
        tr.appendChild(ro(seg.num_tracks ?? '—'));
        tr.appendChild(ro(seg.focus_track_id >= 0 ? seg.focus_track_id : '—'));
        tr.appendChild(importanceTd(seg.focus_importance));
        tr.appendChild(ro(seg.focus_has_face ? '✓' : ''));
        tr.appendChild(ro(seg.focus_area_pct != null ? seg.focus_area_pct.toFixed(1) + '%' : '—'));
        tr.appendChild(txtInput('label', seg.label));
        tr.appendChild(txtInput('subject_focus', seg.subject_focus));
        tr.appendChild(txtInput('description', seg.description));
        tr.appendChild(txtInput('pan_plan', seg.pan_plan));

        const splitTd = document.createElement('td');
        const splitBtn = document.createElement('button');
        splitBtn.className = 'split-btn'; splitBtn.textContent = '✂';
        splitBtn.title = 'Split this segment';
        splitBtn.addEventListener('click', e => { e.stopPropagation(); showSplitInput(splitTd, i); });
        splitTd.appendChild(splitBtn);
        tr.appendChild(splitTd);

        tbody.appendChild(tr);

        // Detail sub-row showing all ranked tracks
        const detailTr = document.createElement('tr');
        detailTr.className = 'detail-row';
        detailTr.id = 'detail-' + i;
        const detailTd = document.createElement('td');
        detailTd.colSpan = 19;
        const tracks = seg.tracks_detail;
        if (!tracks || tracks.length === 0) {
          const p = document.createElement('span');
          p.className = 'detail-no-tracks';
          p.textContent = 'No tracks';
          detailTd.appendChild(p);
        } else {
          const t = document.createElement('table');
          t.className = 'detail-table';
          const hdr = t.insertRow();
          ['rank', 'id', 'importance', 'face', 'area%'].forEach(h => {
            const th = document.createElement('th');
            th.textContent = h;
            hdr.appendChild(th);
          });
          tracks.forEach((tk, rank) => {
            const r = t.insertRow();
            if (rank === 0) r.className = 'focus-track';
            [rank + 1, tk.id, tk.importance.toFixed(3), tk.has_face ? '✓' : '', tk.area_pct.toFixed(1) + '%']
              .forEach(v => { const td = r.insertCell(); td.textContent = v; });
          });
          detailTd.appendChild(t);
        }
        detailTr.appendChild(detailTd);
        tbody.appendChild(detailTr);
      });
    }

    function selectSegment(idx, seekVideo) {
      const wasSelected = selectedIdx === idx;
      selectedIdx = wasSelected ? -1 : idx;
      document.querySelectorAll('.t-seg').forEach((el, i) =>
        el.classList.toggle('selected', i === selectedIdx));
      document.querySelectorAll('#tbody tr.trow').forEach((el, i) =>
        el.classList.toggle('selected', i === selectedIdx));
      // toggle detail row
      document.querySelectorAll('.detail-row').forEach((el, i) =>
        el.classList.toggle('open', i === selectedIdx));
      if (!wasSelected && seekVideo && video.src) {
        video.currentTime = segments[idx].start_s;
      }
      const row = document.getElementById('trow-' + idx);
      if (row) row.scrollIntoView({ block: 'nearest' });
    }

    function showSplitInput(td, idx) {
      const seg = segments[idx];
      const mid = +((seg.start_s + seg.end_s) / 2).toFixed(2);
      td.innerHTML = '';
      const inp = document.createElement('input');
      inp.type = 'number'; inp.step = '0.01'; inp.value = mid; inp.style.width = '68px';
      inp.addEventListener('click', e => e.stopPropagation());
      const ok = document.createElement('button');
      ok.textContent = '✓'; ok.className = 'split-ok-btn';
      ok.addEventListener('click', async e => {
        e.stopPropagation();
        const splitAt = parseFloat(inp.value);
        if (isNaN(splitAt)) return;
        const resp = await fetch('/split', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({segment_index: idx, split_at: splitAt}),
        });
        const data = await resp.json();
        if (data.ok) {
          segments = data.segments;
          renderTimeline();
          renderTable();
          setDirty(true);
          setStatus('Split applied — save to persist', 'ok');
          setTimeout(() => setStatus(''), 3000);
        } else {
          setStatus('Split error: ' + data.error, 'err');
        }
      });
      const cancel = document.createElement('button');
      cancel.textContent = '✕'; cancel.className = 'split-cancel-btn';
      cancel.addEventListener('click', e => { e.stopPropagation(); renderTable(); });
      td.appendChild(inp);
      td.appendChild(ok);
      td.appendChild(cancel);
      inp.focus(); inp.select();
    }

    video.addEventListener('timeupdate', () => {
      const t = video.currentTime;
      let active = -1;
      for (let i = 0; i < segments.length; i++) {
        if (t >= segments[i].start_s && t < segments[i].end_s) { active = i; break; }
      }
      if (active === segments.length - 1 || active === -1) {
        // last segment: include end
        for (let i = segments.length - 1; i >= 0; i--) {
          if (t >= segments[i].start_s) { active = i; break; }
        }
      }
      document.querySelectorAll('.t-seg').forEach((el, i) =>
        el.classList.toggle('active', i === active));
      document.querySelectorAll('#tbody tr.trow').forEach((el, i) =>
        el.classList.toggle('active-row', i === active));
    });

    async function saveData() {
      setStatus('Saving…');
      try {
        const resp = await fetch('/save', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(segments),
        });
        const data = await resp.json();
        if (data.ok) {
          setDirty(false);
          setStatus('Saved ✓', 'ok');
          setTimeout(() => setStatus(''), 3000);
        } else {
          setStatus('Error: ' + data.error, 'err');
        }
      } catch (e) {
        setStatus('Save failed: ' + e.message, 'err');
      }
    }

    document.addEventListener('keydown', e => {
      if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault();
        saveData();
      }
    });

    window.onbeforeunload = () => dirty ? 'Unsaved changes — leave anyway?' : undefined;

    // Init
    fetch('/meta').then(r => r.json()).then(meta => {
      document.getElementById('filename').textContent = meta.filename;
      document.title = 'Plan Editor — ' + meta.filename;
      if (meta.has_video) {
        video.src = '/video';
        video.style.display = 'block';
      } else {
        video.style.display = 'none';
        document.getElementById('no-video').style.display = 'block';
      }
    });

    fetch('/data').then(r => r.json()).then(data => {
      segments = data;
      renderTimeline();
      renderTable();
    });
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return HTML


@app.route("/meta")
def meta():
    return jsonify({
        "filename": Path(STATE["json_path"]).name,
        "has_video": STATE.get("video_path") is not None,
    })


@app.route("/data")
def data():
    with open(STATE["json_path"]) as f:
        return jsonify(json.load(f))


@app.route("/video")
def video():
    path = STATE.get("video_path")
    if not path:
        return Response("No video", status=404)
    return _send_file_range(path)


@app.route("/save", methods=["POST"])
def save():
    segments = request.get_json()
    if not isinstance(segments, list):
        return jsonify({"ok": False, "error": "Expected a JSON array"}), 400

    for seg in segments:
        try:
            if float(seg["start_s"]) >= float(seg["end_s"]):
                return jsonify({
                    "ok": False,
                    "error": f"Segment {seg['segment']}: start_s must be less than end_s"
                }), 400
        except (KeyError, TypeError, ValueError) as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    # Recalculate duration_s server-side
    for seg in segments:
        seg["duration_s"] = round(float(seg["end_s"]) - float(seg["start_s"]), 3)

    json_path = Path(STATE["json_path"])
    tmp_path = json_path.with_suffix(".json.tmp")

    with open(tmp_path, "w") as f:
        json.dump(segments, f, indent=2)
    os.replace(tmp_path, json_path)

    return jsonify({"ok": True})


@app.route("/split", methods=["POST"])
def split():
    body = request.get_json()
    seg_idx = body.get("segment_index")
    split_at = body.get("split_at")

    with open(STATE["json_path"]) as f:
        segments = json.load(f)

    if not isinstance(seg_idx, int) or not (0 <= seg_idx < len(segments)):
        return jsonify({"ok": False, "error": "Invalid segment index"}), 400

    seg = segments[seg_idx]
    start_s = float(seg["start_s"])
    end_s = float(seg["end_s"])

    try:
        split_at = float(split_at)
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "split_at must be a number"}), 400

    if not (start_s < split_at < end_s):
        return jsonify({"ok": False, "error": f"split_at {split_at} must be strictly between {start_s} and {end_s}"}), 400

    first = dict(seg)
    first["end_s"] = round(split_at, 3)
    first["duration_s"] = round(split_at - start_s, 3)

    second = dict(seg)
    second["start_s"] = round(split_at, 3)
    second["duration_s"] = round(end_s - split_at, 3)
    second["trigger"] = "manual_split"
    second["label"] = ""
    second["subject_focus"] = ""
    second["description"] = ""
    second["pan_plan"] = ""

    new_segments = segments[:seg_idx] + [first, second] + segments[seg_idx + 1:]
    for i, s in enumerate(new_segments):
        s["segment"] = i + 1

    return jsonify({"ok": True, "segments": new_segments})


def _send_file_range(path):
    size = os.path.getsize(path)
    range_header = request.headers.get("Range")

    if not range_header:
        def generate_full():
            with open(path, "rb") as f:
                while chunk := f.read(65536):
                    yield chunk
        return Response(generate_full(), mimetype="video/mp4", headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(size),
        })

    byte_range = range_header.replace("bytes=", "")
    parts = byte_range.split("-")
    start = int(parts[0])
    end = int(parts[1]) if parts[1] else size - 1
    end = min(end, size - 1)
    length = end - start + 1

    def generate_range():
        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(65536, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    return Response(generate_range(), status=206, mimetype="video/mp4", headers={
        "Content-Range": f"bytes {start}-{end}/{size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(length),
    })


def _open_browser(port):
    import time
    time.sleep(0.9)
    webbrowser.open(f"http://127.0.0.1:{port}")


def main():
    parser = argparse.ArgumentParser(description="Planning JSON editor — local web UI")
    parser.add_argument("--json", required=True, help="Path to _scenes_planning.json")
    parser.add_argument("--video", default=None, help="Path to debug planning MP4")
    parser.add_argument("--port", type=int, default=5000, help="Local port (default 5000)")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    args = parser.parse_args()

    json_path = Path(args.json).resolve()
    if not json_path.exists():
        print(f"[ERROR] JSON file not found: {json_path}")
        raise SystemExit(1)

    video_path = None
    if args.video:
        video_path = Path(args.video).resolve()
        if not video_path.exists():
            print(f"[WARN]  Video file not found: {video_path} — proceeding without video")
            video_path = None

    STATE["json_path"] = str(json_path)
    STATE["video_path"] = str(video_path) if video_path else None

    print(f"[EDIT]  JSON  : {json_path}")
    if video_path:
        print(f"[EDIT]  Video : {video_path}")
    print(f"[EDIT]  URL   : http://127.0.0.1:{args.port}")

    if not args.no_browser:
        threading.Thread(target=_open_browser, args=(args.port,), daemon=True).start()

    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
