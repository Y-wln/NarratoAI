#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
素材混剪模块：接收N段无声素材+音频文案 → AI理解画面打标签 → 语义匹配 → 混剪成片

工作流：
1. 用Qwen-VL分析每段素材的画面内容（打标签）
2. Whisper转写音频文案
3. DeepSeek做文案→素材语义匹配，生成时间线
4. ffmpeg拼接素材+配音+字幕，自动静音素材原声
"""

import os
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger

from app.utils.qwenvl_analyzer import QwenAnalyzer


class MaterialMixer:
    """素材混剪引擎"""

    def __init__(self, qwen_api_key: str = None, qwen_base_url: str = None,
                 llm_api_key: str = None, llm_base_url: str = None,
                 whisper_model: str = "base", output_dir: str = "./output"):
        self.qwen_api_key = qwen_api_key or os.getenv("DASHSCOPE_API_KEY") or self._read_key("dashscope")
        self.qwen_base_url = qwen_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.llm_api_key = llm_api_key or os.getenv("DEEPSEEK_API_KEY") or self._read_key("deepseek")
        self.llm_base_url = llm_base_url or "https://api.deepseek.com/v1"
        self.whisper_model = whisper_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.analyzer = QwenAnalyzer(
            model_name="qwen-vl-plus",
            api_key=self.qwen_api_key,
            base_url=self.qwen_base_url
        )

    def _read_key(self, provider: str) -> str:
        try:
            import yaml
            cfg = yaml.safe_load(open(os.path.expanduser("~/.hermes/config.yaml")))
            prov = cfg.get("providers", {}).get(provider, {})
            if isinstance(prov, dict):
                return prov.get("api_key", "")
        except Exception:
            pass
        return ""

    # ── 第一步：分析素材画面内容 ──
    def analyze_materials(self, material_paths: List[str]) -> List[Dict]:
        logger.info(f"开始分析 {len(material_paths)} 段素材画面...")
        results = []
        for path in material_paths:
            tags = self._analyze_single_material(path)
            results.append({"path": path, "tags": tags})
            logger.info(f"  {Path(path).name} → {tags[:60]}...")
        return results

    def _analyze_single_material(self, video_path: str) -> str:
        frame_path = "/tmp/_frame_" + Path(video_path).stem + ".jpg"
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", video_path,
                "-vframes", "1", "-q:v", "2", frame_path
            ], capture_output=True, timeout=15, check=True)

            import base64
            with open(frame_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()

            prompt = "请详细描述这个画面：画面里有什么物体、场景、动作、颜色、氛围。用中文关键词回答，以逗号分隔。"
            result = self.analyzer.client.chat.completions.create(
                model="qwen-vl-plus",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": prompt}
                    ]
                }],
                max_tokens=200
            )
            return result.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"分析素材 {video_path} 失败: {e}")
            return "未知画面"
        finally:
            if os.path.exists(frame_path):
                os.remove(frame_path)

    # ── 第二步：转写音频 ──
    def transcribe_audio(self, audio_path: str) -> str:
        logger.info("转写音频文案...")
        try:
            # 先转成wav（whisper需要）
            wav_path = "/tmp/_audio_" + Path(audio_path).stem + ".wav"
            subprocess.run([
                "ffmpeg", "-y", "-i", audio_path,
                "-ar", "16000", "-ac", "1", wav_path
            ], capture_output=True, timeout=30, check=True)

            import whisper
            model = whisper.load_model(self.whisper_model)
            result = model.transcribe(wav_path, language="zh")
            os.remove(wav_path)
            return result.get("text", "").strip()
        except Exception as e:
            logger.error(f"转写失败: {e}")
            return ""

    # ── 第三步：匹配 ──
    def match_materials(self, materials: List[Dict], transcript: str) -> List[Dict]:
        logger.info("LLM匹配素材和文案...")
        from openai import OpenAI

        client = OpenAI(api_key=self.llm_api_key, base_url=self.llm_base_url)

        material_desc = "\n".join([
            f"[素材{i+1}] {m['tags']}" for i, m in enumerate(materials)
        ])

        prompt = f"""你是一个视频剪辑师。以下是多段视频素材的画面描述，和一段文案音频的转写文本。

你的任务：按文案内容，为每一句或每一个语义段落匹配最合适的素材编号。
如果素材不够用，可以重复使用素材。输出JSON格式。

素材描述：
{material_desc}

文案转写：
{transcript}

输出格式（JSON数组）：
[
  {{"text": "文案句子", "material_index": 0, "duration": 5}},
  ...
]
其中material_index从0开始，对应素材列表的索引。duration单位秒。"""

        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=2000
            )
            result = json.loads(resp.choices[0].message.content)
            if isinstance(result, dict):
                for key in ("segments", "timeline", "result"):
                    if key in result:
                        return result[key]
            if isinstance(result, list):
                return result
            return []
        except Exception as e:
            logger.error(f"LLM匹配失败: {e}")
            return []

    # ── 第四步：渲染 ──
    def render(self, timeline: List[Dict], material_paths: List[str],
               audio_path: str, output_name: str = "final.mp4",
               add_transition: bool = True) -> str:
        logger.info("渲染成片...")
        output_path = str(self.output_dir / output_name)

        if add_transition:
            # 构建ffmpeg命令 - 自动静音素材原声+配音
            cmd = ["ffmpeg", "-y"]

            # 先检查素材是否有音频
            has_audio_list = []
            for seg in timeline:
                idx = seg.get("material_index", 0)
                src = material_paths[idx]
                probe = subprocess.run([
                    "ffprobe", "-v", "error", "-select_streams", "a",
                    "-show_entries", "stream=codec_type",
                    "-of", "csv=p=0", src
                ], capture_output=True, text=True, timeout=5)
                has_audio_list.append("audio" in probe.stdout)
                cmd.extend(["-i", src])

            # 加音频输入
            cmd.extend(["-i", audio_path])

            audio_idx = len(timeline)

            # 构建filter
            video_filters = []
            for i in range(len(timeline)):
                dur = timeline[i].get("duration", 5)
                # 静音素材原声，只取视频轨道
                video_filters.append(
                    f"[{i}:v]trim=duration={dur},setpts=PTS-STARTPTS,"
                    f"scale=1080:1920:force_original_aspect_ratio=decrease,"
                    f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}]"
                )

            filter_str = ";".join(video_filters)

            # concat所有视频流
            concat_v = "".join([f"[v{i}]" for i in range(len(timeline))])
            filter_str += f";{concat_v}concat=n={len(timeline)}:v=1:a=0[v_out]"

            # 音频流：直接用用户提供的音频，变速匹配视频时长
            total_video_dur = sum(s.get("duration", 5) for s in timeline)
            audio_dur = self._get_audio_duration(audio_path)
            if audio_dur > 0 and abs(audio_dur - total_video_dur) > 0.5:
                # 变速
                speed = audio_dur / total_video_dur
                filter_str += f";[{audio_idx}:a]atempo={min(2.0, max(0.5, speed))}[a_out]"
            else:
                filter_str += f";[{audio_idx}:a]acopy[a_out]"

            cmd.extend(["-filter_complex", filter_str,
                        "-map", "[v_out]", "-map", "[a_out]",
                        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                        "-c:a", "aac", "-b:a", "128k",
                        "-shortest", output_path])

            try:
                subprocess.run(cmd, check=True, capture_output=True, timeout=600)
                logger.info(f"渲染完成: {output_path}")
                return output_path
            except subprocess.TimeoutExpired:
                logger.error("渲染超时")
                return ""
            except subprocess.CalledProcessError as e:
                logger.error(f"渲染失败: {e.stderr.decode()[:500]}")
                return ""
        else:
            # 无转场，直接用concat demuxer
            concat_file = "/tmp/_concat_list.txt"
            try:
                with open(concat_file, "w") as f:
                    for seg in timeline:
                        idx = seg.get("material_index", 0)
                        dur = seg.get("duration", 5)
                        src = material_paths[idx]
                        f.write(f"file '{src}'\n")
                        f.write(f"duration {dur}\n")

                # 先拼视频
                temp_video = "/tmp/_temp_concat.mp4"
                subprocess.run([
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", concat_file, "-c", "copy", temp_video
                ], check=True, capture_output=True, timeout=300)

                # 再配音
                subprocess.run([
                    "ffmpeg", "-y", "-i", temp_video, "-i", audio_path,
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
                    "-map", "0:v:0", "-map", "1:a:0", "-shortest",
                    output_path
                ], check=True, capture_output=True, timeout=300)

                os.remove(temp_video)
                logger.info(f"渲染完成: {output_path}")
                return output_path
            except Exception as e:
                logger.error(f"渲染失败: {e}")
                return ""
            finally:
                if os.path.exists(concat_file):
                    os.remove(concat_file)

    # ── 一键混剪 ──
    def mix(self, material_paths: List[str], audio_path: str,
            output_name: str = "mixed.mp4", add_transition: bool = True) -> str:
        logger.info(f"=== 开始混剪 ===")
        logger.info(f"素材: {len(material_paths)}段")
        logger.info(f"音频: {audio_path}")

        # Step 1: 素材打标签
        materials = self.analyze_materials(material_paths)

        # Step 2: 转写音频
        transcript = self.transcribe_audio(audio_path)
        logger.info(f"文案: {transcript[:200]}...")

        # Step 3: LLM匹配
        timeline = self.match_materials(materials, transcript)
        if not timeline:
            logger.warning("匹配失败，使用顺序拼接")
            audio_dur = self._get_audio_duration(audio_path)
            seg_dur = audio_dur / len(material_paths) if material_paths else 10
            timeline = [
                {"material_index": i, "duration": seg_dur, "text": ""}
                for i in range(len(material_paths))
            ]

        # Step 4: 渲染
        total = sum(s.get("duration", 5) for s in timeline)
        audio_dur = self._get_audio_duration(audio_path)
        logger.info(f"时间线总时长: {total:.1f}s, 音频时长: {audio_dur:.1f}s")
        output = self.render(timeline, material_paths, audio_path, output_name, add_transition)
        return output

    def _get_audio_duration(self, audio_path: str) -> float:
        try:
            result = subprocess.run([
                "ffprobe", "-v", "error", "-show_entries",
                "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                audio_path
            ], capture_output=True, text=True, timeout=10)
            return float(result.stdout.strip())
        except:
            return 0.0


# ── CLI入口 ──
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("用法: python material_mixer.py <素材目录> <音频文件> [输出文件名]")
        print("示例: python material_mixer.py ./materials/ voice.mp4 mixed.mp4")
        sys.exit(1)

    material_dir = sys.argv[1]
    audio_file = sys.argv[2]
    output_name = sys.argv[3] if len(sys.argv) > 3 else "mixed.mp4"

    exts = (".mp4", ".mov", ".avi", ".mkv", ".webm")
    materials = sorted([
        os.path.join(material_dir, f)
        for f in os.listdir(material_dir)
        if f.lower().endswith(exts)
    ])

    if not materials:
        print(f"素材目录没有视频文件: {material_dir}")
        sys.exit(1)

    print(f"找到 {len(materials)} 段素材")
    mixer = MaterialMixer()
    output = mixer.mix(materials, audio_file, output_name)
    if output:
        print(f"\n✅ 混剪完成: {output}")
    else:
        print("\n❌ 混剪失败")
