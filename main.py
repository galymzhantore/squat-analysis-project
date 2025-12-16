import argparse
from src.video_processor import VideoProcessor


def main():
    parser = argparse.ArgumentParser(description="Squat Analysis Video Processor")
    parser.add_argument("--input", "-i", required=True, help="Input video path")
    parser.add_argument("--output", "-o", help="Output annotated video path")
    parser.add_argument("--json", "-j", help="Output JSON results path")
    parser.add_argument("--side", choices=["left", "right"], default="left",
                        help="Which leg to analyze (default: left)")
    parser.add_argument("--bottom", type=float, default=80.0,
                        help="Bottom threshold angle (default: 80)")
    parser.add_argument("--top", type=float, default=160.0,
                        help="Top threshold angle (default: 160)")
    args = parser.parse_args()
    
    processor = VideoProcessor(
        bottom_threshold=args.bottom,
        top_threshold=args.top
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
