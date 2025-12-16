import argparse
from src.video_processor import VideoProcessor


def main():
    parser = argparse.ArgumentParser(description="Squat Analysis Video Processor")
    parser.add_argument("--input", "-i", required=True, help="Input video path")
    parser.add_argument("--output", "-o", help="Output annotated video path")
    parser.add_argument("--json", "-j", help="Output JSON results path")
    parser.add_argument("--side", choices=["left", "right"], default="left",
                        help="Which leg to analyze (default: left)")
    parser.add_argument("--bottom", type=float, default=90.0,
                        help="Bottom threshold angle (default: 90)")
    parser.add_argument("--rise", type=float, default=20.0,
                        help="Rise threshold for rep count (default: 20)")
    parser.add_argument("--smooth", type=float, default=0.3,
                        help="EMA smoothing factor 0.1-0.9 (lower=smoother, default: 0.3)")
    args = parser.parse_args()
    
    processor = VideoProcessor(
        bottom_threshold=args.bottom,
        rise_threshold=args.rise,
        ema_alpha=args.smooth
    )
    
    print(f"Processing: {args.input}")
    results = processor.process(args.input, args.output, args.side)
    
    final_reps = results[-1]["reps"] if results else 0
    print(f"Total reps: {final_reps}")
    
    if args.json:
        processor.save_results(results, args.json)
        print(f"Results saved: {args.json}")
    
    if args.output:
        print(f"Video saved: {args.output}")


if __name__ == "__main__":
    main()
