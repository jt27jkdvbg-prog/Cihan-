#!/usr/bin/env python3
"""
Otomatik video üretim hattı - Aşama 1

Konu  ->  araştırma  ->  senaryo  ->  görseller  ->  seslendirme  ->  montaj
Çıktı: output/video.mp4 (dikey 1080x1920, altyazılı)
"""
import argparse
import base64
import glob
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# ----------------------------------------------------------------------------
# AYARLAR (istersen GitHub'da "Variables" bölümünden değiştirebilirsin)
# ----------------------------------------------------------------------------
W, H, FPS = 1080, 1920, 30
PAD = 0.35  # sahneler arası kısa sessizlik (saniye)


def _env(name, default):
    return os.environ.get(name) or default


def _models(name, default):
    return [m.strip() for m in _env(name, default).split(",") if m.strip()]


# Gemini (ücretsiz katman). Sırayla dener; biri çalışmazsa sonrakine geçer.
RESEARCH_MODELS = _models("GEMINI_RESEARCH_MODELS", "gemini-2.5-flash,gemini-2.5-flash-lite")
SCRIPT_MODELS = _models("GEMINI_SCRIPT_MODELS", "gemini-3.8-flash,gemini-3.6-flash,gemini-2.5-flash")
# Cloudflare Workers AI (ücretsiz: günde 10.000 nöron)
IMAGE_MODEL = _env("IMAGE_MODEL", "@cf/black-forest-labs/flux-1-schnell")
IMAGE_STEPS = int(_env("IMAGE_STEPS", "6"))
# Ses
TTS_ENGINE = _env("TTS_ENGINE", "edge")  # edge (ücretsiz) | elevenlabs (ücretli, daha kaliteli)
EDGE_VOICE = _env("EDGE_VOICE", "tr-TR-AhmetNeural")  # kadın sesi: tr-TR-EmelNeural
EDGE_RATE = _env("EDGE_RATE", "+0%")
GTTS_LANG = _env("GTTS_LANG", "tr")
ELEVEN_MODEL = _env("TTS_MODEL", "eleven_multilingual_v2")
ELEVEN_VOICE = _env("ELEVEN_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")

TARGET_SECONDS = int(_env("TARGET_SECONDS", "60"))
VIDEO_DILI = _env("VIDEO_DILI", "Türkçe")


def log(msg):
    print(msg, flush=True)


def retry(fn, what, tries=3, wait=6):
    for i in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            log(f"  ! {what} başarısız (deneme {i}/{tries}): {e}")
            if i == tries:
                raise
            time.sleep(wait * i)


def run(cmd, cwd=None):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("Komut hatası:\n" + " ".join(map(str, cmd[:6])) + " ...\n" + r.stderr[-1800:])
    return r


# ----------------------------------------------------------------------------
# 1) ARAŞTIRMA + SENARYO (Gemini - ücretsiz katman)
# ----------------------------------------------------------------------------
RESEARCH_PROMPT = """Konu: {topic}

Bu konuda derin bir araştırma yap. Web araması yapabiliyorsan mutlaka kullan. Aşağıda Vikipedi'den alınmış ek kaynak metni de var; onu da değerlendir.
Kısa ve etkileyici bir video için kullanılacak DOĞRULANMIŞ bilgileri topla: şaşırtıcı gerçekler, rakamlar, tarihler, ilginç hikâyeler, yaygın yanlış inanışlar, dramatik anlar.
Birden fazla güvenilir kaynakla teyit edemediğin veya emin olmadığın bilgiyi "DOĞRULANAMADI" diye işaretle. Hiçbir şey uydurma.
Çıktı: Türkçe, madde madde araştırma notları. En sonda kullandığın kaynakların listesi.

EK KAYNAK (Vikipedi):
{ctx}"""

SCRIPT_PROMPT = """Aşağıdaki araştırma notlarına dayanarak "{topic}" konusunda dikey formatta (YouTube Shorts / TikTok / Reels) KISA FİLM tadında bir video hazırla.

KURALLAR:
- Anlatım dili: {dil}. Toplam anlatım yaklaşık {kelime} kelime olsun (yaklaşık {sure} saniye).
- {n_min}-{n_max} sahne. Her sahnenin anlatımı 1-3 cümle.
- Yapı: İlk sahnede merak uyandıran güçlü bir açılış (izleyiciyi ilk 3 saniyede yakala). Ortada gerilim ve gelişme. Sonda etkileyici bir kapanış.
- SADECE notlardaki doğrulanmış bilgileri kullan. "DOĞRULANAMADI" işaretli veya emin olmadığın hiçbir şeyi yazma, uydurma.
- Anlatım metni sesli okunacak: rakamları ve kısaltmaları okunduğu gibi yaz; emoji, parantez, sahne yönergesi koyma.
- "style": TÜM görsellerde ortak kullanılacak İNGİLİZCE görsel stil tarifi (ışık, renk paleti, sinematik üslup, lens, atmosfer). Tekrar eden bir karakter varsa görünüşünü burada ayrıntılı tarif et.
- Her sahne için "image_prompt": İNGİLİZCE, o sahneyi anlatan somut, sinematik görsel tarifi. Ana özne kadrajın MERKEZİNDE olsun (görsel sonra dikey kırpılacak). Görselde yazı, harf, logo OLMASIN. Gerçek ve tanınabilir kişilerin yüzü OLMASIN (isim verme; silüet, arkadan görünüm, eller, nesneler, manzara, sembolik sahneler kullan).
- "title": en fazla 70 karakter, merak uyandıran başlık. "description": 2-3 cümlelik açıklama + 3-5 hashtag. "tags": 8-12 etiket.

ARAŞTIRMA NOTLARI:
{notes}

SADECE geçerli JSON döndür, başka hiçbir şey yazma:
{{"title": "...", "description": "...", "tags": ["..."], "style": "...", "scenes": [{{"narration": "...", "image_prompt": "..."}}]}}"""

UA = {"User-Agent": "video-fabrikasi/1.0 (kisisel proje)"}


def gemini_generate(prompt, models, search=False, json_mode=False, max_tokens=8192):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    last = None
    for m in models:
        try:
            cfg = {"max_output_tokens": max_tokens, "temperature": 0.8}
            if search:
                cfg["tools"] = [types.Tool(google_search=types.GoogleSearch())]
            if json_mode:
                cfg["response_mime_type"] = "application/json"
            resp = client.models.generate_content(
                model=m, contents=prompt, config=types.GenerateContentConfig(**cfg)
            )
            text = (resp.text or "").strip()
            if text:
                log(f"     (model: {m}{', web aramalı' if search else ''})")
                return text
            last = RuntimeError(f"{m} boş yanıt verdi")
        except Exception as e:  # noqa: BLE001
            last = e
            log(f"     ! {m} çalışmadı: {str(e)[:160]}")
    raise RuntimeError(f"Hiçbir Gemini modeli çalışmadı. Son hata: {last}")


def wiki_context(topic, per_lang_chars=3500):
    """Ücretsiz ek kaynak: Vikipedi (Türkçe + İngilizce) ilk sonuç."""
    import requests

    parts = []
    for lang in ("tr", "en"):
        try:
            api = f"https://{lang}.wikipedia.org/w/api.php"
            r = requests.get(
                api,
                params={"action": "query", "list": "search", "srsearch": topic, "format": "json", "srlimit": 1},
                headers=UA, timeout=30,
            )
            hits = r.json()["query"]["search"]
            if not hits:
                continue
            title = hits[0]["title"]
            r2 = requests.get(
                api,
                params={"action": "query", "prop": "extracts", "explaintext": 1, "exlimit": 1,
                        "titles": title, "redirects": 1, "format": "json"},
                headers=UA, timeout=30,
            )
            page = next(iter(r2.json()["query"]["pages"].values()))
            text = (page.get("extract") or "").strip()
            if text:
                parts.append(f"[Vikipedi/{lang}: {title}]\n{text[:per_lang_chars]}")
        except Exception as e:  # noqa: BLE001
            log(f"     (Vikipedi/{lang} alınamadı: {str(e)[:80]})")
    return "\n\n".join(parts)


def research(topic):
    ctx = wiki_context(topic)
    prompt = RESEARCH_PROMPT.replace("{topic}", topic).replace("{ctx}", ctx or "(Vikipedi'de sonuç bulunamadı)")
    try:
        notes = gemini_generate(prompt, RESEARCH_MODELS, search=True, max_tokens=6000)
    except Exception as e:  # noqa: BLE001
        log(f"     web aramalı araştırma çalışmadı, Vikipedi ile devam ediliyor ({str(e)[:100]})")
        notes = gemini_generate(prompt, SCRIPT_MODELS, search=False, max_tokens=6000)
    if len(notes) < 200:
        raise RuntimeError("Araştırma sonucu çok kısa geldi.")
    return notes


def parse_json(text):
    a, b = text.find("{"), text.rfind("}")
    if a == -1 or b == -1:
        raise ValueError("Yanıtta JSON bulunamadı")
    return json.loads(text[a : b + 1])


def write_script(topic, notes):
    n = max(6, round(TARGET_SECONDS / 5))
    prompt = (
        SCRIPT_PROMPT.replace("{topic}", topic)
        .replace("{dil}", VIDEO_DILI)
        .replace("{kelime}", str(int(TARGET_SECONDS * 2.4)))
        .replace("{sure}", str(TARGET_SECONDS))
        .replace("{n_min}", str(n - 1))
        .replace("{n_max}", str(n + 2))
        .replace("{notes}", notes)
    )
    text = gemini_generate(prompt, SCRIPT_MODELS, json_mode=True, max_tokens=8192)
    data = parse_json(text)
    scenes = data.get("scenes") or []
    if len(scenes) < 3 or not data.get("style"):
        raise ValueError("Senaryo eksik geldi")
    for s in scenes:
        if not s.get("narration") or not s.get("image_prompt"):
            raise ValueError("Bir sahnede anlatım veya görsel tarifi yok")
    data["title"] = (data.get("title") or topic)[:95]
    return data


# ----------------------------------------------------------------------------
# 2) GÖRSELLER (Cloudflare Workers AI - FLUX schnell, ücretsiz kota)
# ----------------------------------------------------------------------------
def make_image(prompt, path):
    import requests

    acc, tok = os.environ["CF_ACCOUNT_ID"], os.environ["CF_API_TOKEN"]
    r = requests.post(
        f"https://api.cloudflare.com/client/v4/accounts/{acc}/ai/run/{IMAGE_MODEL}",
        headers={"Authorization": f"Bearer {tok}"},
        json={"prompt": prompt[:2000], "steps": IMAGE_STEPS},
        timeout=180,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Cloudflare hata {r.status_code}: {r.text[:300]}")
    if r.headers.get("content-type", "").startswith("image/"):
        data = r.content
    else:
        j = r.json()
        img = (j.get("result") or {}).get("image")
        if not img:
            raise RuntimeError(f"Cloudflare görsel döndürmedi: {str(j)[:300]}")
        data = base64.b64decode(img)
    if len(data) < 5000:
        raise RuntimeError("Görsel çok küçük/bozuk geldi")
    Path(path).write_bytes(data)


# ----------------------------------------------------------------------------
# 3) SESLENDİRME - kelime zamanlamalarıyla
#    Varsayılan: edge-tts (ücretsiz). Yedek: gTTS. İsteğe bağlı: ElevenLabs.
# ----------------------------------------------------------------------------
_state = {"engine": TTS_ENGINE}


def estimate_words(text, total):
    """Kelime zamanları gelmezse, konuşma süresine orantılı tahmin."""
    ws = text.split()
    if not ws:
        return []
    lens = [len(w) + 1 for w in ws]
    tot, t, avail, out = sum(lens), 0.05, max(total - 0.1, 0.5), []
    for w, ln in zip(ws, lens):
        d = avail * ln / tot
        out.append((w, t, t + d))
        t += d
    return out


def make_voice_edge(text, mp3_path):
    import asyncio

    import edge_tts

    async def go():
        try:
            com = edge_tts.Communicate(text, EDGE_VOICE, rate=EDGE_RATE, boundary="WordBoundary")
        except TypeError:  # eski sürüm
            com = edge_tts.Communicate(text, EDGE_VOICE, rate=EDGE_RATE)
        audio, words = bytearray(), []
        async for ch in com.stream():
            if ch["type"] == "audio":
                audio.extend(ch["data"])
            elif ch["type"] == "WordBoundary":
                s = ch["offset"] / 1e7
                words.append((ch["text"], s, s + ch["duration"] / 1e7))
        return bytes(audio), words

    audio, words = asyncio.run(go())
    if len(audio) < 1000:
        raise RuntimeError("edge-tts ses döndürmedi")
    Path(mp3_path).write_bytes(audio)
    return words or estimate_words(text, audio_duration(mp3_path))


def make_voice_gtts(text, mp3_path):
    from gtts import gTTS

    gTTS(text, lang=GTTS_LANG).save(str(mp3_path))
    return estimate_words(text, audio_duration(mp3_path))


def words_from_alignment(al):
    words, cur, ws, we = [], "", None, None
    for c, s, e in zip(
        al["characters"], al["character_start_times_seconds"], al["character_end_times_seconds"]
    ):
        if c.isspace():
            if cur:
                words.append((cur, ws, we))
                cur, ws = "", None
        else:
            if not cur:
                ws = s
            cur += c
            we = e
    if cur:
        words.append((cur, ws, we))
    return words


def make_voice_eleven(text, mp3_path, prev_text="", next_text=""):
    import requests

    url = (
        f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE}/with-timestamps"
        "?output_format=mp3_44100_128"
    )
    body = {
        "text": text,
        "model_id": ELEVEN_MODEL,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75, "style": 0.25, "use_speaker_boost": True},
    }
    if "v3" not in ELEVEN_MODEL:
        if prev_text:
            body["previous_text"] = prev_text[-300:]
        if next_text:
            body["next_text"] = next_text[:300]
    r = requests.post(
        url,
        headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"], "Content-Type": "application/json"},
        json=body,
        timeout=180,
    )
    if r.status_code != 200:
        raise RuntimeError(f"ElevenLabs hata {r.status_code}: {r.text[:300]}")
    j = r.json()
    Path(mp3_path).write_bytes(base64.b64decode(j["audio_base64"]))
    return words_from_alignment(j["alignment"])


def choose_voice_engine(work):
    """Videonun tamamı aynı sesle çıksın diye motoru en başta yoklar."""
    if _state["engine"] == "elevenlabs":
        return
    try:
        make_voice_edge("Merhaba, bu bir deneme.", Path(work) / "probe.mp3")
        _state["engine"] = "edge"
    except Exception as e:  # noqa: BLE001
        log(f"     ! edge-tts çalışmadı ({str(e)[:140]}). Yedek ses (gTTS) kullanılacak.")
        _state["engine"] = "gtts"
    log(f"     ses motoru: {_state['engine']}")


def make_voice(text, mp3_path, prev_text="", next_text=""):
    eng = _state["engine"]
    if eng == "elevenlabs":
        return make_voice_eleven(text, mp3_path, prev_text, next_text)
    if eng == "gtts":
        return make_voice_gtts(text, mp3_path)
    return make_voice_edge(text, mp3_path)


# ----------------------------------------------------------------------------
# 4) MONTAJ (FFmpeg)
# ----------------------------------------------------------------------------
def audio_duration(path):
    r = run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)]
    )
    return float(r.stdout.strip())


def make_clip(image, frames, idx, out):
    """Duran görsele sinematik kamera hareketi (yavaş zoom / kaydırma) verir."""
    d = frames
    variants = [
        ("1+0.18*on/{d}", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"),  # yavaş yakınlaş
        ("1.18-0.18*on/{d}", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"),  # yavaş uzaklaş
        ("1.2", "(iw-iw/zoom)*on/{d}", "ih/2-(ih/zoom/2)"),  # soldan sağa
        ("1.2", "(iw-iw/zoom)*(1-on/{d})", "ih/2-(ih/zoom/2)"),  # sağdan sola
    ]
    z, x, y = [v.replace("{d}", str(d)) for v in variants[idx % len(variants)]]
    vf = (
        f"scale={W*2}:{H*2}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={W*2}:{H*2},"
        f"zoompan=z='{z}':x='{x}':y='{y}':d={d}:s={W}x{H}:fps={FPS},"
        "format=yuv420p"
    )
    run(
        [
            "ffmpeg", "-y", "-i", str(image), "-vf", vf, "-frames:v", str(d),
            "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-r", str(FPS), str(out),
        ]
    )


def caption_events(words, offset, max_words=3):
    events, chunk = [], []

    def flush():
        if chunk:
            events.append([offset + chunk[0][1], offset + chunk[-1][2], " ".join(w[0] for w in chunk)])
            chunk.clear()

    for w in words:
        chunk.append(w)
        if len(chunk) >= max_words or re.search(r"[.!?…:;]$", w[0]) or (
            len(chunk) >= 2 and w[0].endswith(",")
        ):
            flush()
    flush()
    for i in range(len(events) - 1):
        events[i][1] = min(events[i + 1][0], events[i][1] + 0.35)
    if events:
        events[-1][1] += 0.25
    return events


def ass_time(t):
    cs = int(round(max(t, 0) * 100))
    return f"{cs // 360000}:{cs % 360000 // 6000:02d}:{cs % 6000 // 100:02d}.{cs % 100:02d}"


def write_ass(events, path):
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,78,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,7,2,2,90,90,430,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    for s, e, text in events:
        clean = re.sub(r"[{}\\]", "", text).strip()
        lines.append(f"Dialogue: 0,{ass_time(s)},{ass_time(e)},Default,,0,0,0,,{clean}")
    Path(path).write_text(header + "\n".join(lines) + "\n", encoding="utf-8")


def assemble(items, work, out_path, music=None):
    """items: [{'image': Path, 'audio': Path, 'words': [(kelime, bas, bit), ...]}, ...]"""
    work = Path(work).resolve()
    out_path = Path(out_path).resolve()
    t, events = 0.0, []
    for i, it in enumerate(items):
        dur = audio_duration(it["audio"])
        frames = int(round((dur + PAD) * FPS))
        seg = frames / FPS
        log(f"  sahne {i + 1}/{len(items)}: {seg:.1f} sn")
        make_clip(it["image"], frames, i, work / f"clip_{i:03d}.mp4")
        run(
            [
                "ffmpeg", "-y", "-i", str(it["audio"]), "-af", "apad", "-ar", "44100", "-ac", "2",
                "-t", f"{seg:.4f}", str(work / f"nar_{i:03d}.wav"),
            ]
        )
        events += caption_events(it["words"], t)
        t += seg
    total = t

    (work / "clips.txt").write_text("".join(f"file 'clip_{i:03d}.mp4'\n" for i in range(len(items))))
    (work / "nars.txt").write_text("".join(f"file 'nar_{i:03d}.wav'\n" for i in range(len(items))))
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "clips.txt", "-c", "copy", "video_raw.mp4"], cwd=work)
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "nars.txt", "-c", "copy", "narration.wav"], cwd=work)
    write_ass(events, work / "subs.ass")

    cmd = ["ffmpeg", "-y", "-i", "video_raw.mp4", "-i", "narration.wav"]
    if music:
        cmd += ["-stream_loop", "-1", "-i", str(Path(music).resolve())]
        fade_out_at = max(total - 2.5, 0)
        fc = (
            "[0:v]ass=subs.ass[v];"
            f"[2:a]atrim=0:{total:.3f},asetpts=PTS-STARTPTS,volume=0.10,"
            f"afade=t=in:d=1.5,afade=t=out:st={fade_out_at:.3f}:d=2.5[m];"
            "[1:a][m]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[a]"
        )
    else:
        fc = "[0:v]ass=subs.ass[v];[1:a]anull[a]"
    cmd += [
        "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-t", f"{total:.3f}",
        str(out_path),
    ]
    run(cmd, cwd=work)
    return total


# ----------------------------------------------------------------------------
# ANA AKIŞ
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True, help="Video konusu")
    topic = ap.parse_args().topic.strip()

    need = ["GEMINI_API_KEY", "CF_ACCOUNT_ID", "CF_API_TOKEN"]
    if TTS_ENGINE == "elevenlabs":
        need.append("ELEVENLABS_API_KEY")
    for k in need:
        if not os.environ.get(k):
            sys.exit(f"HATA: {k} bulunamadı. GitHub > Settings > Secrets and variables > Actions bölümüne ekle.")

    out, work = Path("output"), Path("work")
    out.mkdir(exist_ok=True)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()

    log(f"Konu: {topic}")
    log("1/5  Araştırma yapılıyor...")
    notes = retry(lambda: research(topic), "araştırma")
    (out / "research.md").write_text(notes, encoding="utf-8")

    log("2/5  Senaryo yazılıyor...")
    script = retry(lambda: write_script(topic, notes), "senaryo")
    (out / "script.json").write_text(json.dumps(script, ensure_ascii=False, indent=2), encoding="utf-8")
    scenes = script["scenes"]
    log(f"     Başlık: {script['title']}  |  {len(scenes)} sahne")

    log("3/5  Görseller üretiliyor...")
    style = script["style"].strip().rstrip(".")
    prompts = [
        f"{style}. {s['image_prompt'].strip()} Main subject centered in the frame, simple uncluttered "
        "background on both sides, cinematic, highly detailed, no text, no letters, no watermark."
        for s in scenes
    ]
    img_paths = [work / f"img_{i:03d}.jpg" for i in range(len(scenes))]

    def do_image(i):
        retry(lambda: make_image(prompts[i], img_paths[i]), f"görsel {i + 1}")
        log(f"     görsel {i + 1}/{len(scenes)} hazır")

    with ThreadPoolExecutor(max_workers=3) as ex:
        list(ex.map(do_image, range(len(scenes))))

    log("4/5  Seslendirme yapılıyor...")
    choose_voice_engine(work)
    items = []
    for i, s in enumerate(scenes):
        mp3 = work / f"voice_{i:03d}.mp3"
        prev_t = scenes[i - 1]["narration"] if i > 0 else ""
        next_t = scenes[i + 1]["narration"] if i < len(scenes) - 1 else ""
        words = retry(lambda: make_voice(s["narration"], mp3, prev_t, next_t), f"ses {i + 1}")
        items.append({"image": img_paths[i], "audio": mp3, "words": words})
        log(f"     ses {i + 1}/{len(scenes)} hazır")

    log("5/5  Montaj yapılıyor...")
    tracks = glob.glob("assets/music/*.mp3")
    music = random.choice(tracks) if tracks else None
    log(f"     müzik: {music or 'yok (assets/music klasörüne .mp3 eklenebilir)'}")
    total = assemble(items, work, out / "video.mp4", music)

    shutil.copy(img_paths[0], out / "thumbnail.jpg")
    meta = {
        "topic": topic,
        "title": script["title"],
        "description": script.get("description", ""),
        "tags": script.get("tags", []),
        "duration_seconds": round(total, 1),
    }
    (out / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"BİTTİ: output/video.mp4  ({total:.0f} sn)")


if __name__ == "__main__":
    main()
