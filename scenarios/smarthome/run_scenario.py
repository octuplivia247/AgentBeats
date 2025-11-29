#!/usr/bin/env python3
"""
Smart Home Scenario Runner

Runs the smart home evaluation scenario following official AgentBeats patterns.
"""

import asyncio
import sys
import argparse
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.launcher import launch_evaluation_refactored


async def run_smart_home_scenario(
    show_logs: bool = False,
    serve_only: bool = False,
    scenario_file: str = "scenarios/smarthome/scenario.toml"
):
    """
    Run the smart home evaluation scenario.
    
    Args:
        show_logs: Show agent outputs during assessment
        serve_only: Start agents without running assessment
        scenario_file: Path to scenario TOML file
    """
    print("=" * 70)
    print("AgentBeats - Smart Home Evaluation Scenario")
    print("=" * 70)
    print(f"Scenario: {scenario_file}")
    print(f"Show logs: {show_logs}")
    print(f"Serve only: {serve_only}")
    print("=" * 70)
    
    if serve_only:
        print("\n🚀 Starting agents in serve-only mode...")
        print("Agents will remain running. Press Ctrl+C to stop.")
        print("\nTo run assessment manually, use:")
        print("  agentbeats-client scenarios/smarthome/scenario.toml")
        print()
    
    # Run the evaluation
    await launch_evaluation_refactored()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run Smart Home evaluation scenario"
    )
    parser.add_argument(
        "--show-logs",
        action="store_true",
        help="Show agent outputs during assessment"
    )
    parser.add_argument(
        "--serve-only",
        action="store_true",
        help="Start agents without running assessment"
    )
    parser.add_argument(
        "--scenario",
        default="scenarios/smarthome/scenario.toml",
        help="Path to scenario TOML file"
    )
    
    args = parser.parse_args()
    
    asyncio.run(run_smart_home_scenario(
        show_logs=args.show_logs,
        serve_only=args.serve_only,
        scenario_file=args.scenario
    ))


if __name__ == "__main__":
    main()

