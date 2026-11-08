#!/usr/bin/env python3
"""
Otomatik video üretim hattı - Aşama 2 (hareketli klipler)

Konu -> araştırma -> senaryo -> ana kareler (FLUX) -> hareketli klipler (Boreal)
     -> seslendirme -> montaj (altyazı + ses efekti + müzik) -> output/video.mp4
"""
import argparse
import base64
import glob
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from cinematic_director import (
    build_script_prompt,
    enrich_image_prompt,
    resolve_visual_style,
    validate_script,
)
from quality_gate import require_real_motion, validate_motion_outputs
from camera_director import build_zoompan_filter

# ----------------------------------------------------------------------------
# AYARLAR (istersen GitHub'da "Variables" bölümünden değiştirebilirsin)
# ----------------------------------------------------------------------------
W, H, FPS = 1080, 1920, 30
PAD = 0.4  # sahneler arası kısa nefes payı (saniye)


def _env(name, default):
    return os.environ.get(name) or default


def _list(name, default):
    return [m.strip() for m in _env(name, default).split(",") if m.strip()]


# Gemini (ücretsiz katman). Sırayla dener; biri çalışmazsa sonrakine geçer.
RESEARCH_MODELS = _list("GEMINI_RESEARCH_MODELS", "gemini-3.8-flash,gemini-3.6-flash")
SCRIPT_MODELS = _list("GEMINI_SCRIPT_MODELS", "gemini-3.8-flash,gemini-3.6-flash,gemini-3.5-flash-lite")
GEMINI_TIMEOUT_MS = int(_env("GEMINI_TIMEOUT_SEC", "90")) * 1000

# Görsel motoru: auto = FAL_KEY varsa fal.ai (dikey, daha kaliteli), yoksa Cloudflare (ücretsiz)
IMAGE_ENGINE = _env("IMAGE_ENGINE", "auto")  # auto | fal | cloudflare
CF_IMAGE_MODEL = _env("CF_IMAGE_MODEL", "@cf/black-forest-labs/flux-1-schnell")
CF_STEPS = int(_env("CF_STEPS", "6"))
# Hareketli klip: auto = FAL_KEY varsa üretir (ücretli), yoksa kamera hareketli ücretsiz mod
MOTION = _env("MOTION", "auto")  # auto | off

# fal.ai: ana kare (görsel) ve hareketli klip
FAL_IMAGE_MODEL = _env("FAL_IMAGE_MODEL", "fal-ai/flux/dev")
VIDEO_MODEL = _env("VIDEO_MODEL", "creatify/boreal")
VIDEO_RES = _env("VIDEO_RES", "720p")
AMBIENCE_VOLUME = float(_env("AMBIENCE_VOLUME", "0.3"))  # kliplerin kendi ses efekti seviyesi

# Ses motorları sırayla denenir: gemini (en doğal), edge (ücretsiz), gtts (yedek), elevenlabs (ücretli)
TTS_ENGINES = _list("TTS_ENGINES", "gemini,edge,gtts")
GEMINI_TTS_MODEL = _env("GEMINI_TTS_MODEL", "gemini-3.8-flash-tts")
GEMINI_VOICE = _env("GEMINI_VOICE", "Charon")
GEMINI_TTS_STYLE = _env(
    "GEMINI_TTS_STYLE",
    "Read the following in Turkish like a cinematic documentary narrator: calm, deep, "
    "natural pacing, clear pronunciation, subtle emotion",
)
EDGE_VOICE = _env("EDGE_VOICE", "tr-TR-AhmetNeural")  # kadın sesi: tr-TR-EmelNeural
EDGE_RATE = _env("EDGE_RATE", "+0%")
GTTS_LANG = _env("GTTS_LANG", "tr")
ELEVEN_MODEL = _env("TTS_MODEL", "eleven_multilingual_v2")
ELEVEN_VOICE = _env("ELEVEN_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")

TARGET_SECONDS = int(_env("TARGET_SECONDS", "45"))
VIDEO_DILI = _env("VIDEO_DILI", "Türkçe")
VISUAL_STYLE = _env("VISUAL_STYLE", "3d")
CUSTOM_VISUAL_STYLE = _env("CUSTOM_VISUAL_STYLE", "")
NEGATIVE = (
    "subtitles, text, letters, watermark, logo, "
    "distorted face, extra fingers, morphing, flicker, "
    "deformed hands, unnatural anatomy, duplicate people"
)


def log(msg):
    print(msg, flush=True)


def retry(fn, what, tries=3, wait=6):
    for i in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            log(f"  ! {what} başarısız (deneme {i}/{tries}): {str(e)[:300]}")
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

SCRIPT_PROMPT = """Aşağıdaki araştırma notlarına dayanarak "{topic}" konusunda dikey formatta (YouTube Shorts / TikTok / Reels) KISA FİLM tadında bir video hazırla. Görseller yapay zekâ ile üretilip hareketlendirilecek.

KURALLAR:
- Anlatım dili: {dil}. Toplam anlatım yaklaşık {kelime} kelime olsun (yaklaşık {sure} saniye).
- {n_min}-{n_max} sahne. Her sahnenin anlatımı 1-2 kısa cümle (en fazla ~12 saniye okunacak kadar).
- Yapı: İlk sahnede merak uyandıran güçlü bir açılış (izleyiciyi ilk 3 saniyede yakala). Ortada gerilim ve gelişme. Sonda etkileyici bir kapanış.
- SADECE notlardaki doğrulanmış bilgileri kullan. "DOĞRULANAMADI" işaretli veya emin olmadığın hiçbir şeyi yazma, uydurma.
- Anlatım metni sesli okunacak: rakamları ve kısaltmaları okunduğu gibi yaz; emoji, parantez, sahne yönergesi koyma.
- "style": TÜM sahnelerde ortak kullanılacak İNGİLİZCE görsel stil tarifi. Varsayılan olarak fotogerçekçi sinema estetiği (ışık, renk paleti, lens, atmosfer, film greni). Tekrar eden karakter varsa yaşını, saçını, kıyafetini, yüz özelliklerini burada AYRINTILI ve sabit biçimde tarif et ki her sahnede aynı kişi görünsün.
- Her sahne için:
  * "image_prompt": İNGİLİZCE, sahnenin İLK KARESİNİ anlatan somut sinematik görsel tarifi (kadraj, özne, mekân, ışık). Görselde yazı, harf, logo OLMASIN.
  * "motion_prompt": İNGİLİZCE, o karede NE HAREKET EDİYOR: karakterin doğal ve gerçekçi hareketi (yürümek, dönmek, nefes almak, bakmak, el hareketi) ve KAMERA hareketi (yavaş dolly-in, yan kayma, hafif el kamerası vb.). Tek, net, sade bir hareket yaz. Karakter konuşmasın.
  * "sound": İNGİLİZCE, sadece ortam sesi ve ses efekti (rüzgâr, adım sesi, dalga, uğultu...). Müzik ve konuşma YAZMA.
- Gerçek ve tanınabilir kişilerin yüzü OLMASIN (isim verme; silüet, arkadan görünüm, anonim karakterler, nesneler, manzara, sembolik sahneler kullan).
- "title": en fazla 70 karakter, merak uyandıran başlık. "description": 2-3 cümlelik açıklama + 3-5 hashtag. "tags": 8-12 etiket.

ARAŞTIRMA NOTLARI:
{notes}

SADECE geçerli JSON döndür, başka hiçbir şey yazma:
{{"title": "...", "description": "...", "tags": ["..."], "style": "...", "scenes": [{{"narration": "...", "image_prompt": "...", "motion_prompt": "...", "sound": "..."}}]}}"""

UA = {"User-Agent": "video-fabrikasi/2.0 (kisisel proje)"}


def _gemini_client():
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS),
    )
    return client, types


def gemini_generate(prompt, models, search=False, json_mode=False, max_tokens=8192):
    client, types = _gemini_client()
    last = None
    for m in models:
        for attempt in (1, 2):
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
                break
            except Exception as e:  # noqa: BLE001
                last = e
                msg = str(e)
                log(f"     ! {m} çalışmadı: {msg[:160]}")
                busy = any(k in msg for k in ("503", "429", "UNAVAILABLE", "timed out", "Timeout", "timeout"))
                if busy and attempt == 1:
                    time.sleep(8)
                    continue
                break
    raise RuntimeError(f"Hiçbir Gemini modeli çalışmadı. Son hata: {str(last)[:200]}")


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
    prompt = build_script_prompt(
        topic,
        notes,
        TARGET_SECONDS,
        VIDEO_DILI,
        VISUAL_STYLE,
        CUSTOM_VISUAL_STYLE
    )

    raw = gemini_generate(
        prompt,
        SCRIPT_MODELS,
        json_mode=True,
        max_tokens=8192
    )

    data = parse_json(raw)
    data["style"] = resolve_visual_style(VISUAL_STYLE, CUSTOM_VISUAL_STYLE)
    

    report = validate_script(data, TARGET_SECONDS)

    if not report["valid"]:
        raise ValueError(
            "Senaryo kalite kontrolünden geçemedi: "
            + "; ".join(report["errors"])
        )

    character_bible = data.get("character_bible", "")

    for scene in data["scenes"]:
        scene.setdefault(
            "motion_prompt",
            "The subject performs a natural physical action while the camera tracks smoothly."
        )
        scene.setdefault(
            "sound",
            "Subtle location ambience and realistic physical sound effects."
        )

    data["title"] = (data.get("title") or topic)[:95]
    data["quality_report"] = report

    return data



# ----------------------------------------------------------------------------
# 2) ANA KARELER (fal.ai - FLUX) ve 3) HAREKETLİ KLİPLER (fal.ai - Boreal)
# ----------------------------------------------------------------------------
def _download(url, path, min_bytes):
    import requests

    data = requests.get(url, timeout=300).content
    if len(data) < min_bytes:
        raise RuntimeError(f"İndirilen dosya çok küçük/bozuk ({len(data)} bayt)")
    Path(path).write_bytes(data)


def make_keyframe(prompt, path):
    import fal_client

    result = fal_client.subscribe(
        FAL_IMAGE_MODEL,
        arguments={
            "prompt": prompt,
            "image_size": {"width": 720, "height": 1280},  # ~1 megapiksel
            "num_images": 1,
            "enable_safety_checker": True,
            "output_format": "jpeg",
        },
    )
    flags = result.get("has_nsfw_concepts") or []
    if flags and flags[0]:
        raise RuntimeError("Güvenlik filtresi görseli engelledi, yeniden denenecek")
    _download(result["images"][0]["url"], path, 5000)


def make_keyframe_cloudflare(prompt, path):
    import requests

    acc, tok = os.environ["CF_ACCOUNT_ID"], os.environ["CF_API_TOKEN"]
    r = requests.post(
        f"https://api.cloudflare.com/client/v4/accounts/{acc}/ai/run/{CF_IMAGE_MODEL}",
        headers={"Authorization": f"Bearer {tok}"},
        json={"prompt": prompt[:2000], "steps": CF_STEPS},
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


def pick_image_engine():
    eng = IMAGE_ENGINE
    if eng == "auto":
        eng = "fal" if os.environ.get("FAL_KEY") else "cloudflare"
    return eng


def motion_text(style, scene):
    dialogue = str(scene.get("dialogue", "")).strip()

    if dialogue:
        speech_prompt = (
            "The character visibly speaks naturally with realistic lip movement. "
            f"Intended dialogue: {dialogue}. "
            "Match facial expressions and mouth movement to the spoken line."
        )
    else:
        speech_prompt = (
            "No required dialogue. Use natural facial expressions "
            "and realistic movement."
        )

    return (
        f"[VISUAL]\n{style}. "
        f"{scene['motion_prompt'].strip()} ""High-quality animated feature-film movement matching the selected "
"visual style. Animate the character's body, face, clothing and "
"environment with clear physical actions. Use smooth cinematic camera "
"movement. Preserve character identity, proportions and clothing.\n\n"
        f"[SPEECH]\n{speech_prompt}\n\n"
        f"[SOUNDS]\n{scene['sound'].strip()}. "
        "Natural environmental sound effects.\n\n"
        "[TEXT]\nNo subtitles, captions, logos or written text."
    )
def make_motion(image_path, prompt, duration, out_path):
    import fal_client

    image_url = fal_client.upload_file(str(image_path))
    result = fal_client.subscribe(
        VIDEO_MODEL,
        arguments={
            "prompt": prompt,
            "image_url": image_url,
            "resolution": VIDEO_RES,
            "aspect_ratio": "9:16",
            "duration": duration,
            "negative_prompt": NEGATIVE,
            "enable_safety_checker": True,
        },
    )
    _download(result["video"]["url"], out_path, 20000)


# ----------------------------------------------------------------------------
# 4) SESLENDİRME - motorlar sırayla denenir, tek videoda tek ses kullanılır
# ----------------------------------------------------------------------------
def audio_duration(path):
    r = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)])
    return float(r.stdout.strip())


def estimate_words(text, total):
    """Kelime zamanları gelmezse, konuşma süresine orantılı tahmin (noktalama duraklamalı)."""
    ws = text.split()
    if not ws:
        return []
    weights = []
    for w in ws:
        wt = len(w) + 1
        if re.search(r"[.!?…]$", w):
            wt += 6
        elif re.search(r"[,;:]$", w):
            wt += 3
        weights.append(wt)
    tot, t, avail, out = sum(weights), 0.03, max(total - 0.06, 0.5), []
    for w, wt in zip(ws, weights):
        d = avail * wt / tot
        out.append((w, t, t + d))
        t += d
    return out


def _pcm_to_mp3(pcm, rate, mp3_path):
    raw = Path(str(mp3_path) + ".pcm")
    raw.write_bytes(pcm)
    trim = (
        "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.05,"
        "areverse,silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.05,areverse"
    )
    run(
        ["ffmpeg", "-y", "-f", "s16le", "-ar", str(rate), "-ac", "1", "-i", str(raw),
         "-af", trim, "-ar", "44100", "-c:a", "libmp3lame", "-b:a", "192k", str(mp3_path)]
    )
    raw.unlink(missing_ok=True)


def make_voice_gemini(text, mp3_path, prev_text="", next_text=""):
    client, types = _gemini_client()
    last = None
    for attempt in (1, 2, 3):
        try:
            resp = client.models.generate_content(
                model=GEMINI_TTS_MODEL,
                contents=f"{GEMINI_TTS_STYLE}: {text}",
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=GEMINI_VOICE)
                        )
                    ),
                ),
            )
            part = resp.candidates[0].content.parts[0]
            data = part.inline_data.data
            if isinstance(data, str):
                data = base64.b64decode(data)
            mime = getattr(part.inline_data, "mime_type", "") or ""
            m = re.search(r"rate=(\d+)", mime)
            rate = int(m.group(1)) if m else 24000
            if len(data) < 2000:
                raise RuntimeError("Gemini TTS ses döndürmedi")
            _pcm_to_mp3(data, rate, mp3_path)
            return estimate_words(text, audio_duration(mp3_path))
        except Exception as e:  # noqa: BLE001
            last = e
            msg = str(e)
            if attempt < 3 and any(k in msg for k in ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE")):
                log(f"     (Gemini TTS yoğun, {35} sn bekleniyor)")
                time.sleep(35)
                continue
            break
    raise RuntimeError(f"Gemini TTS: {str(last)[:200]}")


def make_voice_edge(text, mp3_path, prev_text="", next_text=""):
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


def make_voice_gtts(text, mp3_path, prev_text="", next_text=""):
    from gtts import gTTS

    gTTS(text, lang=GTTS_LANG).save(str(mp3_path))
    return estimate_words(text, audio_duration(mp3_path))


def words_from_alignment(al):
    words, cur, ws, we = [], "", None, None
    for c, s, e in zip(al["characters"], al["character_start_times_seconds"], al["character_end_times_seconds"]):
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


def make_voice_elevenlabs(text, mp3_path, prev_text="", next_text=""):
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


VOICE_FUNCS = {
    "gemini": make_voice_gemini,
    "edge": make_voice_edge,
    "gtts": make_voice_gtts,
    "elevenlabs": make_voice_elevenlabs,
}


def synthesize_all(scenes, work):
    """Her motoru sırayla dener; tüm sahneler aynı motorla üretilir (ses tutarlı olsun diye)."""
    last = None
    for eng in TTS_ENGINES:
        fn = VOICE_FUNCS.get(eng)
        if not fn:
            continue
        if eng == "elevenlabs" and not os.environ.get("ELEVENLABS_API_KEY"):
            continue
        try:
            log(f"     ses motoru: {eng}")
            results = []
            for i, s in enumerate(scenes):
                mp3 = Path(work) / f"voice_{i:03d}.mp3"
                prev_t = scenes[i - 1]["narration"] if i > 0 else ""
                next_t = scenes[i + 1]["narration"] if i < len(scenes) - 1 else ""
                words = retry(lambda: fn(s["narration"], mp3, prev_t, next_t), f"ses {i + 1}", tries=2, wait=4)
                results.append((mp3, words))
                log(f"     ses {i + 1}/{len(scenes)} hazır")
            return results
        except Exception as e:  # noqa: BLE001
            last = e
            log(f"     ! '{eng}' ses motoru çalışmadı ({str(e)[:140]}), sıradaki deneniyor")
    raise RuntimeError(f"Hiçbir ses motoru çalışmadı. Son hata: {str(last)[:200]}")


# ----------------------------------------------------------------------------
# 5) MONTAJ (FFmpeg)
# ----------------------------------------------------------------------------def make_clip(image, frames, idx, out, scene=None):
    """Ücretsiz modda sahneye göre sinematik kamera hareketi uygular."""
    d = max(1, int(frames))

    camera_filter = build_zoompan_filter(
        scene or {},
        idx,
        d,
        W,
        H,
        FPS,
    )

    vf = (
        f"scale={W*2}:{H*2}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={W*2}:{H*2},"
        f"{camera_filter},"
        "eq=contrast=1.06:saturation=1.12,"
        "vignette=PI/5,noise=alls=6:allf=t+u,"
        "format=yuv420p"
    )

    run(
        [
            "ffmpeg", "-y", "-i", str(image),
            "-vf", vf,
            "-frames:v", str(d),
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "20",
            "-r", str(FPS),
            str(out),
        ]
    )
    run(
        ["ffmpeg", "-y", "-i", str(image), "-vf", vf, "-frames:v", str(d),
         "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-r", str(FPS), str(out)]
    )


def norm_clip(src, frames, out):
    """Yapay zekâ klibini 1080x1920 / 30 fps'e getirir; kısa kalırsa son kareyi uzatır."""
    vf = (
        f"scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,crop={W}:{H},setsar=1,"
        f"fps={FPS},tpad=stop_mode=clone:stop_duration=4,unsharp=3:3:0.4,format=yuv420p"
    )
    run(
        ["ffmpeg", "-y", "-i", str(src), "-vf", vf, "-an", "-frames:v", str(frames),
         "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-r", str(FPS), str(out)]
    )


def has_audio(path):
    r = run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", str(path)])
    return bool(r.stdout.strip())


def ambience(src, seg, out):
    """Klibin kendi ses efektini (rüzgâr, adım sesi vb.) kısık seviyede alır."""
    if src and has_audio(src):
        fade_out = max(seg - 0.3, 0)
        af = f"volume={AMBIENCE_VOLUME},afade=t=in:d=0.15,afade=t=out:st={fade_out:.3f}:d=0.3,apad"
        run(["ffmpeg", "-y", "-i", str(src), "-vn", "-ac", "2", "-ar", "44100", "-af", af, "-t", f"{seg:.4f}", str(out)])
    else:
        run(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", f"{seg:.4f}", str(out)])


def caption_events(words, offset, max_words=3):
    events, chunk = [], []

    def flush():
        if chunk:
            events.append([offset + chunk[0][1], offset + chunk[-1][2], " ".join(w[0] for w in chunk)])
            chunk.clear()

    for w in words:
        chunk.append(w)
        if len(chunk) >= max_words or re.search(r"[.!?…:;]$", w[0]) or (len(chunk) >= 2 and w[0].endswith(",")):
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
    """items: [{'image': Path, 'clip': Path|None, 'audio': Path, 'words': [(kelime, bas, bit)]}, ...]"""
    work = Path(work).resolve()
    out_path = Path(out_path).resolve()
    t, events = 0.0, []
    for i, it in enumerate(items):
        dur = audio_duration(it["audio"])
        frames = int(round((dur + PAD) * FPS))
        seg = frames / FPS
        log(f"  sahne {i + 1}/{len(items)}: {seg:.1f} sn {'(hareketli klip)' if it.get('clip') else '(yedek: kamera hareketi)'}")
        clip_out = work / f"clip_{i:03d}.mp4"
        if it.get("clip"):
            norm_clip(it["clip"], frames, clip_out)
        else:
    make_clip(
        it["image"],
        frames,
        i,
        clip_out,
        scene=it.get("scene", {}),
    )
        run(
            ["ffmpeg", "-y", "-i", str(it["audio"]), "-af", "apad", "-ar", "44100", "-ac", "2",
             "-t", f"{seg:.4f}", str(work / f"nar_{i:03d}.wav")]
        )
        events += caption_events(it["words"], t)
        t += seg
    total = t

    n = len(items)
    (work / "clips.txt").write_text("".join(f"file 'clip_{i:03d}.mp4'\n" for i in range(n)))
    (work / "nars.txt").write_text("".join(f"file 'nar_{i:03d}.wav'\n" for i in range(n)))
    (work / "ambs.txt").write_text("".join(f"file 'amb_{i:03d}.wav'\n" for i in range(n)))
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "clips.txt", "-c", "copy", "video_raw.mp4"], cwd=work)
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "nars.txt", "-c", "copy", "narration.wav"], cwd=work)
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "ambs.txt", "-c", "copy", "ambience.wav"], cwd=work)
    write_ass(events, work / "subs.ass")

    cmd = ["ffmpeg", "-y", "-i", "video_raw.mp4", "-i", "narration.wav", "-i", "ambience.wav"]
    fc = (
        "[0:v]ass=subs.ass[v];"
        "[1:a][2:a]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[nm];"
    )
    if music:
        cmd += ["-stream_loop", "-1", "-i", str(Path(music).resolve())]
        fade_out_at = max(total - 2.5, 0)
        fc += (
            f"[3:a]atrim=0:{total:.3f},asetpts=PTS-STARTPTS,volume=0.10,"
            f"afade=t=in:d=1.5,afade=t=out:st={fade_out_at:.3f}:d=2.5[m];"
            "[nm][m]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[mix];"
        )
    else:
        fc += "[nm]anull[mix];"
    fc += "[mix]loudnorm=I=-16:TP=-1.5:LRA=11[a]"
    cmd += [
        "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-movflags", "+faststart", "-t", f"{total:.3f}",
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

    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("HATA: GEMINI_API_KEY bulunamadı. GitHub > Settings > Secrets and variables > Actions bölümüne ekle.")
    image_engine = pick_image_engine()
    if image_engine == "fal" and not os.environ.get("FAL_KEY"):
        sys.exit("HATA: FAL_KEY bulunamadı.")
    if image_engine == "cloudflare" and not (os.environ.get("CF_ACCOUNT_ID") and os.environ.get("CF_API_TOKEN")):
        sys.exit("HATA: CF_ACCOUNT_ID ve CF_API_TOKEN secret'ları bulunamadı (ücretsiz görsel için gerekli).")
    motion_on = MOTION != "off" and bool(os.environ.get("FAL_KEY"))

    out, work = Path("output"), Path("work")
    out.mkdir(exist_ok=True)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()

    log(f"Konu: {topic}  |  hedef süre: {TARGET_SECONDS} sn")
    log(f"Mod: görsel={image_engine}, hareketli klip={'AÇIK (ücretli)' if motion_on else 'KAPALI (ücretsiz mod)'}")
    log("1/6  Araştırma yapılıyor...")
    notes = retry(lambda: research(topic), "araştırma", tries=2)
    (out / "research.md").write_text(notes, encoding="utf-8")

    log("2/6  Senaryo yazılıyor...")
    script = retry(lambda: write_script(topic, notes), "senaryo", tries=3)
    (out / "script.json").write_text(json.dumps(script, ensure_ascii=False, indent=2), encoding="utf-8")
    scenes = script["scenes"]
    log(f"     Başlık: {script['title']}  |  {len(scenes)} sahne")

    log("3/6  Ana kareler üretiliyor...")
    style = script["style"].strip().rstrip(".")

    center = (
        " Main subject centered in the frame, simple uncluttered background on both sides."
        if image_engine == "cloudflare" else ""
    )
    prompts = [
        (
            enrich_image_prompt(
                style,
                script.get("character_bible", ""),
                s
            )
            + center
            + " Vertical 9:16 composition, premium cinematic animated film frame, "
"highly detailed environment, consistent selected visual style, "
"no text, no letters, no watermark."
        )
        for s in scenes
    ]
    keyframe_fn = make_keyframe if image_engine == "fal" else make_keyframe_cloudflare
    img_paths = [work / f"img_{i:03d}.jpg" for i in range(len(scenes))]

    def do_image(i):
        retry(lambda: keyframe_fn(prompts[i], img_paths[i]), f"görsel {i + 1}")
        log(f"     ana kare {i + 1}/{len(scenes)} hazır")

    with ThreadPoolExecutor(max_workers=3) as ex:
        list(ex.map(do_image, range(len(scenes))))

    log("4/6  Seslendirme yapılıyor...")
    voices = synthesize_all(scenes, work)

    clip_paths = [work / f"motion_{i:03d}.mp4" for i in range(len(scenes))]
    ok = [False] * len(scenes)
    if motion_on:
        log("5/6  Hareketli klipler üretiliyor (bu adım birkaç dakika sürer)...")
        durations = []
        for mp3, _ in voices:
            durations.append(int(min(20, max(3, math.ceil(audio_duration(mp3) + PAD)))))
        est = len(scenes) * (0.025 if image_engine == "fal" else 0) + sum(durations) * 0.01
        log(f"     tahmini fal.ai maliyeti: yaklaşık ${est:.2f}")

        def do_motion(i):
            try:
                prompt = motion_text(style, scenes[i])
                retry(lambda: make_motion(img_paths[i], prompt, durations[i], clip_paths[i]), f"klip {i + 1}", tries=3, wait=10)
                ok[i] = True
                log(f"     klip {i + 1}/{len(scenes)} hazır")
            except Exception as e:  # noqa: BLE001
                log(f"     ! klip {i + 1} üretilemedi, yedek kamera hareketi kullanılacak: {str(e)[:200]}")

        with ThreadPoolExecutor(max_workers=4) as ex:
            list(ex.map(do_motion, range(len(scenes))))

        if ok.count(False) > len(scenes) / 2:
            sys.exit(
                "HATA: Hareketli kliplerin çoğu üretilemedi. fal.ai bakiyeni (fal.ai/dashboard/billing) ve "
                "yukarıdaki hata satırlarını kontrol et. Slayt gibi video yayınlamamak için işlem durduruldu."
            )
    else:
        log("5/6  Hareketli klip atlandı (ücretsiz mod: sinematik kamera hareketi kullanılacak)")

    log("6/6  Montaj yapılıyor...")
    tracks = glob.glob("assets/music/*.mp3")
    music = random.choice(tracks) if tracks else None
    log(f"     müzik: {music or 'yok (assets/music klasörüne .mp3 eklenebilir)'}")
    items = [
        {
            "image": img_paths[i],
            "clip": clip_paths[i] if ok[i] else None,
            "audio": voices[i][0],
            "words": voices[i][1],
        }
        for i in range(len(scenes))
    ]
    total = assemble(items, work, out / "video.mp4", music)

    shutil.copy(img_paths[0], out / "thumbnail.jpg")
    meta = {
        "topic": topic,
        "title": script["title"],
        "description": script.get("description", ""),
        "tags": script.get("tags", []),
        "duration_seconds": round(total, 1),
        "motion_clips": f"{sum(ok)}/{len(scenes)}",
    }
    (out / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"BİTTİ: output/video.mp4  ({total:.0f} sn, hareketli klip: {meta['motion_clips']})")


if __name__ == "__main__":
    main()

