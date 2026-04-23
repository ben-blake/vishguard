"""Command-line entry point: python -m vishguard.cli run <audio> --out <dir>."""
from __future__ import annotations

import argparse
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="vishguard",
        description="VishGuard — phone-call audio risk pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Analyze a phone-call recording")
    run_p.add_argument("audio", type=Path, help="Path to WAV/MP3/M4A file")
    run_p.add_argument("--out", type=Path, default=Path("out"), help="Output directory")
    run_p.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"], help="Compute device"
    )
    run_p.add_argument("--no-tts", action="store_true", help="Skip TTS briefing stage")
    run_p.add_argument(
        "--whisper", default="openai/whisper-small", help="Whisper model ID"
    )
    run_p.add_argument(
        "--prompt", default="v4", choices=["v1", "v2", "v3", "v4"], help="Tactic prompt variant"
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.command == "run":
        from .orchestrator import runPipeline
        from .types import AsrConfig, LlmConfig, SpoofConfig, TtsConfig

        report = runPipeline(
            audioPath=args.audio,
            outDir=args.out,
            asrCfg=AsrConfig(modelId=args.whisper, device=args.device),
            spoofCfg=SpoofConfig(device=args.device),
            llmCfg=LlmConfig(device=args.device, promptVariant=args.prompt),
            ttsCfg=TtsConfig(device=args.device),
            runTts=not args.no_tts,
        )

        print(f"Risk score : {report.risk.score}/100 ({report.risk.band.upper()})")
        print(f"Reasoning  : {report.risk.reasoning}")
        print(f"Tactics    : {', '.join(t.label for t in report.tactics) or 'none'}")
        print(f"Report     : {args.out / 'report.json'}")
        if report.briefingAudioPath:
            print(f"Briefing   : {report.briefingAudioPath}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
