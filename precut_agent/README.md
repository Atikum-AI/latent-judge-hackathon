# Pre-cut agent (OpenNegative Lab, Meta hackathon)

Turns a shoot day's camera files into a take list the director approves before anything is encoded.
Code measures; Muse Spark decides; every decision is logged with its reason. Non-destructive: cuts and sync are manifest
entries only, originals are never trimmed, renamed or deleted.

Steps (`./precut.sh <step> shoot_<date>`; `./precut.sh all` runs ingest, takes, sync, dialogue, label, plan, report):
1. `ingest`: ffprobe (timecode, fps from `r_frame_rate`, size, bit depth, colour tags) and a SHA-256 per original.
   Files that are not 24.000 fps are flagged.
2. `sync`: external sound to the camera's scratch audio by clap cross-correlation (timecode first), with a drift check
   at the end of long takes.
3. `takes`: take boundaries from clap spikes and the spoken slate. "action", "cut" and "circle that" come from Meta's
   transcription model, with times taken from speech onsets.
4. `dialogue`: what is said between "action" and "cut", line by line, with a start time and a speaker
   (`person_1`, `person_2` ..., never names). Muse Voice Transcribe (`muse-voice-transcribe-1.0`) is tried first;
   while Meta does not serve it on our account (404, checked 29 Oct) `muse-wonder-1.0` transcribes, and the take
   list names the model that actually ran. The audio is the external recorder when sync matched it (the
   drift-corrected copy if there is one) and it lines up with the camera's audio of the take, else the
   scratch audio (`--scratch` forces it). The code drops crew calls, clamps times, marks lines no speech onset
   corroborates, drops every line when the clip holds no speech, and flags text that looks like a full name.
   `--replay` reuses `precut_cache/dialogue.json` with no API call; `--no-transcribe` skips the step. The label
   step uses these words for `labels.dialogue`, and the report adds a Dialogue section and `dialogue/<id>.srt`.
5. `label`: Muse Spark 1.3 labels scene / shot / take, dialogue, a best-take guess (a label, never a circle) and flags;
   `--require-spark` refuses to fall back. Spark also gets 3 small key frames of the usable range (`--frames N`,
   `--frames 0` for text only) and writes `labels.picture`: shot type, camera move, subjects, action, location,
   lighting, picture issues, a usable guess and which frames each claim came from. The code keeps only known values,
   names people `person_1`, `person_2` ... (anything that looks like a real name is replaced and logged) and drops
   evidence pointing at frames that were not sent. `--replay` reuses the answers saved in
   `precut_cache/spark_labels.json` with no API call. Key frames are cached in `precut_cache/keyframes/`.
6. `plan`: the encode plan, 241-frame windows every 192 frames at 1280x704.
7. `report`: the take list (markdown + JSON) for approval, with a "picture (Spark)" column and the dialogue per take.

`cloud_watch.py` picks up takes synced from Blackmagic Cloud (proxy first, then the original replaces it).

Tests: `pytest -q` (Meta calls stubbed, no network, no cost). The test clips are synthetic (`make_test_fixtures.py`); no real
footage, no transcripts of real people are in this repo. Requires ffmpeg/ffprobe; optional PySceneDetect for camera stops.
Keys come from the environment (`META_API_KEY`, `META_BASE_URL`), never from files.
