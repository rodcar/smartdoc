#!/usr/bin/env python3
"""
Independent HTTP Endpoint Accuracy Test

This test acts as an external client using the requests library to test
the document analysis endpoint. It samples 100 images from the validation
dataset and calculates accuracy percentage.

Usage:
    python tests/test_endpoint_accuracy.py

Requirements:
    - Django server must be running on localhost:8000
    - Validation dataset must exist in output/docs-sm_validation/
    - requests library must be installed
"""

import os
import sys
import glob
import random
import json
import time
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import requests
except ImportError:
    print("❌ Error: requests library not found. Please install it with: pip install requests")
    sys.exit(1)

# Configuration
BASE_URL = "http://localhost:8000"
ANALYZE_ENDPOINT = f"{BASE_URL}/api/analyze/"
VALIDATION_DATASET_PATH = project_root / "output" / "docs-sm_test"
NUM_SAMPLES = None  # Set to None to use all images, or set to a number to sample
RANDOM_SEED = 42

# Document types (should match Django settings)
DOCUMENT_TYPES = [
    "advertisement", "budget", "email", "file_folder", "form", 
    "handwritten", "invoice", "letter", "memo", "news_article", 
    "presentation", "questionnaire", "resume", "scientific_publication", 
    "scientific_report", "specification"
]


class EndpointAccuracyTester:
    """Independent HTTP endpoint accuracy tester using requests library."""
    
    def __init__(self):
        """Initialize the tester with configuration."""
        self.base_url = BASE_URL
        self.analyze_endpoint = ANALYZE_ENDPOINT
        self.validation_path = VALIDATION_DATASET_PATH
        self.num_samples = NUM_SAMPLES
        self.document_types = DOCUMENT_TYPES
        
        # Verify validation dataset exists
        if not self.validation_path.exists():
            raise FileNotFoundError(f"Validation dataset not found at: {self.validation_path}")
    
    def collect_validation_images(self) -> List[Tuple[str, str]]:
        """
        Collect all validation images with their true document types.
        
        Returns:
            List of tuples (image_path, true_document_type)
        """
        validation_images = []
        
        for doc_type in self.document_types:
            doc_type_path = self.validation_path / doc_type
            if doc_type_path.exists():
                # Get all .jpg files in this document type folder
                jpg_files = list(doc_type_path.glob("*.jpg"))
                for jpg_file in jpg_files:
                    validation_images.append((str(jpg_file), doc_type))
        
        return validation_images
    
    def get_sample_images(self, validation_images: List[Tuple[str, str]], 
                         num_samples: int = None) -> List[Tuple[str, str]]:
        """
        Get a random sample of validation images.
        
        Args:
            validation_images: List of all validation images
            num_samples: Number of images to sample (default: self.num_samples)
            
        Returns:
            List of tuples (image_path, true_document_type)
        """
        if num_samples is None:
            num_samples = self.num_samples
            
        # If num_samples is None (use all images) or if we have fewer images than requested
        if num_samples is None or len(validation_images) <= num_samples:
            return validation_images
        
        # Use random seed for reproducible results
        random.seed(RANDOM_SEED)
        return random.sample(validation_images, num_samples)
    
    def test_server_connection(self) -> bool:
        """
        Test if the server is running and accessible.
        
        Returns:
            True if server is accessible, False otherwise
        """
        try:
            response = requests.get(self.base_url, timeout=5)
            return True
        except requests.exceptions.RequestException as e:
            print(f"❌ Server connection failed: {e}")
            print(f"💡 Make sure Django server is running on {self.base_url}")
            print("   Run: python manage.py runserver")
            return False
    
    def analyze_document(self, image_path: str) -> Optional[Dict]:
        """
        Send HTTP request to analyze endpoint.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Response data as dict or None if request failed
        """
        try:
            with open(image_path, 'rb') as f:
                files = {'image': (os.path.basename(image_path), f, 'image/jpeg')}
                
                response = requests.post(
                    self.analyze_endpoint,
                    files=files,
                    timeout=600  # 30 second timeout
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    print(f"❌ Request failed with status {response.status_code}: {response.text}")
                    return None
                    
        except requests.exceptions.RequestException as e:
            print(f"❌ Request exception: {e}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return None
    
    def run_accuracy_test(self) -> Dict:
        """
        Run the main accuracy test.
        
        Returns:
            Dictionary with test results
        """
        print("🧪 HTTP Endpoint Accuracy Test")
        print("=" * 60)
        print(f"🌐 Testing endpoint: {self.analyze_endpoint}")
        print(f"📁 Dataset path: {self.validation_path}")
        print(f"🎯 Document types: {len(self.document_types)} types")
        print()
        
        # Test server connection
        if not self.test_server_connection():
            return {"error": "Server connection failed"}
        
        # Collect validation images
        print("📂 Collecting validation images...")
        validation_images = self.collect_validation_images()
        
        if not validation_images:
            return {"error": "No validation images found"}
        
        print(f"📊 Found {len(validation_images)} total validation images")
        
        # Get sample for testing
        sample_images = self.get_sample_images(validation_images)
        total_images = len(sample_images)
        
        if self.num_samples is None:
            print(f"🎯 Using all {total_images} images for testing")
        else:
            print(f"🎲 Sampling {total_images} images for testing")
        print()
        
        # Track results
        correct_predictions = 0
        failed_requests = 0
        results_by_type = {doc_type: {"correct": 0, "total": 0} for doc_type in self.document_types}
        detailed_results = []
        
        # Process each image
        start_time = time.time()
        
        for i, (image_path, true_type) in enumerate(sample_images, 1):
            print(f"📷 Processing {i}/{total_images}: {os.path.basename(image_path)}", end=" ... ")
            
            # Send request to analyze endpoint
            result = self.analyze_document(image_path)
            
            if result:
                predicted_type = result.get('document_type', 'unknown')
                
                # Check if prediction is correct
                is_correct = predicted_type == true_type
                if is_correct:
                    correct_predictions += 1
                    print("✅")
                else:
                    print(f"❌ (predicted: {predicted_type}, true: {true_type})")
                
                # Update per-type statistics
                results_by_type[true_type]["total"] += 1
                if is_correct:
                    results_by_type[true_type]["correct"] += 1
                
                # Store detailed result
                detailed_results.append({
                    "image": os.path.basename(image_path),
                    "true_type": true_type,
                    "predicted_type": predicted_type,
                    "correct": is_correct,
                    "entities": len(result.get('entities', []))
                })
                
            else:
                failed_requests += 1
                print("❌ Request failed")
            
            # Progress indicator every 10 images
            if i % 10 == 0:
                current_accuracy = (correct_predictions / (i - failed_requests)) * 100 if (i - failed_requests) > 0 else 0
                elapsed_time = time.time() - start_time
                avg_time_per_request = elapsed_time / i
                print(f"📈 Progress: {i}/{total_images} - Current Accuracy: {current_accuracy:.1f}% - Avg Time: {avg_time_per_request:.2f}s")
        
        # Calculate final results
        successful_requests = total_images - failed_requests
        if successful_requests > 0:
            accuracy_percentage = (correct_predictions / successful_requests) * 100
        else:
            accuracy_percentage = 0.0
        
        total_time = time.time() - start_time
        
        # Print detailed results
        print("\n" + "=" * 60)
        print("📊 HTTP ENDPOINT ACCURACY TEST RESULTS")
        print("=" * 60)
        print(f"📁 Total images tested: {total_images}")
        print(f"✅ Successful requests: {successful_requests}")
        print(f"❌ Failed requests: {failed_requests}")
        print(f"🎯 Correct predictions: {correct_predictions}")
        print(f"📈 Overall Accuracy: {accuracy_percentage:.2f}%")
        print(f"⏱️  Total time: {total_time:.2f}s")
        print(f"⚡ Average time per request: {total_time/total_images:.2f}s")
        print()
        
        # Print per-type accuracy
        print("📋 Accuracy by Document Type:")
        print("-" * 50)
        for doc_type in sorted(self.document_types):
            type_stats = results_by_type[doc_type]
            if type_stats["total"] > 0:
                type_accuracy = (type_stats["correct"] / type_stats["total"]) * 100
                print(f"{doc_type:25}: {type_accuracy:5.1f}% ({type_stats['correct']:2d}/{type_stats['total']:2d})")
            else:
                print(f"{doc_type:25}: No samples")
        
        # Print some example predictions
        print("\n📝 Sample Predictions:")
        print("-" * 70)
        print(f"{'Status':<6} {'Image':<25} {'True Type':<20} {'Predicted':<20} {'Entities':<8}")
        print("-" * 70)
        for result in detailed_results[:15]:  # Show first 15 results
            status = "✅" if result["correct"] else "❌"
            print(f"{status:<6} {result['image'][:24]:<25} {result['true_type']:<20} {result['predicted_type']:<20} {result['entities']:<8}")
        
        if len(detailed_results) > 15:
            print(f"... and {len(detailed_results) - 15} more results")
        
        print("\n" + "=" * 60)
        
        # Final assessment
        if accuracy_percentage >= 70.0:
            print(f"🎉 EXCELLENT: Accuracy {accuracy_percentage:.2f}% meets the ≥70% target!")
        elif accuracy_percentage >= 50.0:
            print(f"✅ GOOD: Accuracy {accuracy_percentage:.2f}% is above 50%")
        else:
            print(f"⚠️  IMPROVEMENT NEEDED: Accuracy {accuracy_percentage:.2f}% is below 50%")
        
        print("=" * 60)
        
        return {
            "total_images": total_images,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "correct_predictions": correct_predictions,
            "accuracy_percentage": accuracy_percentage,
            "total_time": total_time,
            "avg_time_per_request": total_time / total_images,
            "results_by_type": results_by_type,
            "detailed_results": detailed_results
        }


def main():
    """Main function to run the accuracy test."""
    try:
        print("🚀 Starting HTTP Endpoint Accuracy Test")
        print(f"📍 Working directory: {os.getcwd()}")
        print(f"📁 Project root: {project_root}")
        print()
        
        # Create and run tester
        tester = EndpointAccuracyTester()
        results = tester.run_accuracy_test()
        
        # Check for errors
        if "error" in results:
            print(f"❌ Test failed: {results['error']}")
            return 1
        
        # Save results to file
        results_file = project_root / "output" / "endpoint_accuracy_results.json"
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {results_file}")
        
        # Return appropriate exit code
        accuracy = results.get("accuracy_percentage", 0)
        if accuracy >= 70.0:
            return 0  # Success
        else:
            return 1  # Below target accuracy
            
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 