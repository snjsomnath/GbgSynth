#!/usr/bin/env python3
"""Script to generate neighbourhood heights for missing areas."""

import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    from gbgsynth.data_utils import (
        generate_neighbourhood_heights,
        get_missing_neighbourhood_heights
    )
    
    # First show what's missing
    missing = get_missing_neighbourhood_heights()
    print(f"\n{'='*60}")
    print(f"Found {len(missing)} missing neighbourhoods to process")
    print(f"{'='*60}\n")
    
    if not missing:
        print("All neighbourhoods already have height files!")
        return 0
    
    for i, m in enumerate(missing, 1):
        print(f"  {i:2d}. {m['code']}: {m['name']}")
    
    print(f"\n{'='*60}")
    print("Starting processing...")
    print("This will download pointcloud data and compute heights.")
    print("Each neighbourhood may take several minutes.")
    print(f"{'='*60}\n")
    
    try:
        result = generate_neighbourhood_heights()
        
        print(f"\n{'='*60}")
        if result:
            print("Processing completed successfully!")
        else:
            print("Processing completed with some failures.")
        print(f"{'='*60}\n")
        
        return 0 if result else 1
        
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user.")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
