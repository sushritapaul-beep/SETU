# Recording the demo GIF

A 15-second GIF at the top of the README is the highest-leverage thing in the
repo: most people who open it will never run the code, and a recruiter scanning
on a phone will watch the loop and read nothing else. It has to show a Hindi
question producing a cited answer.

## What to capture

Show the three things that make SETU different, in this order:

1. **A Devanagari question being typed** — proves multilingual input is real.
2. **The progress trail** — `Detecting language → Translating your question →
   Searching scheme documents → …`. This is the agent visibly working, and it
   fills the seconds while the pipeline runs.
3. **The answer with its Sources panel expanded** — proves the citations are
   real links to myScheme, not decoration.

If an eligibility query fits in the time budget, prefer it: the ✅/❌ verdict
panel is the most distinctive thing in the product.

## Setup before recording

```powershell
# Terminal 1
docker compose up -d
uvicorn api.main:app --reload

# Terminal 2
python ui/app.py
```

**Warm the models first.** Ask one throwaway question and wait for it to finish.
The first query downloads and loads ~6 GB; recording a cold start would produce
45 seconds of nothing.

Then:
- Set the browser to **1280×800** and zoom to ~110% so text is readable when the
  GIF is scaled down in the README.
- Use light mode. Dark screenshots lose detail after GIF colour quantisation.
- Clear the chat so the recording starts on an empty conversation.
- Close other apps — the pipeline needs headroom, and a memory stall mid-record
  wastes the take.

## The query to use

```
ZED प्रमाणन के लिए कौन से दस्तावेज़ आवश्यक हैं?
```

Paste it rather than typing character by character; typing Devanagari on camera
is slow and eats the budget. Expect roughly:

| Segment | Seconds |
|---|---|
| Query appears, send | 0–2 |
| Progress trail advancing | 2–9 |
| Answer renders | 9–12 |
| Click Sources, panel expands | 12–15 |

## Tools

| Platform | Tool | Notes |
|---|---|---|
| Windows | [ScreenToGif](https://www.screentogif.com/) | Best option — records, trims and exports in one app |
| Windows | Xbox Game Bar (`Win+G`) → GIF converter | Records MP4; needs a second step |
| macOS | [Kap](https://getkap.co/) | Exports GIF directly |
| Any | OBS → `ffmpeg` | Most control, most steps |

If you record MP4 first:

```bash
ffmpeg -i demo.mp4 -vf "fps=12,scale=900:-1:flags=lanczos,split[s0][s1];\
[s0]palettegen[p];[s1][p]paletteuse" -loop 0 docs/demo.gif
```

## Constraints

- **Under 15 seconds** and **under 5 MB.** GitHub will render a larger GIF, but
  it loads slowly and looks unprofessional on mobile.
- **12 fps is enough.** This is UI, not motion — halving frame rate roughly
  halves file size with no perceptible loss.
- **900 px wide.** Matches GitHub's README column; wider just gets downscaled.
- **Crop to the chat panel.** Your bookmarks bar and OS taskbar are not part of
  the demo.
- **No API keys on screen.** Check the terminal is not visible and no `.env` tab
  is open in the editor.

## Where it goes

Save as `docs/demo.gif`, then replace the placeholder near the top of
`README.md`:

```markdown
<!-- Replace this line once docs/demo.gif exists -->
<p align="center"><img src="docs/demo.gif" alt="SETU answering a Hindi query" width="900"></p>
```

Commit it normally — a few MB is fine in git. Only reach for Git LFS if you end
up with several videos.
