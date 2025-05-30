#!/usr/bin/env python3
"""
Simple test script for fuzzywuzzy-first entity recognition
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ml.entity_recognition import BrandRecognizer

def test_fuzzywuzzy_approach():
    """Test the fuzzywuzzy-first approach with various examples."""
    
    print("🔍 Testing Fuzzywuzzy-First Entity Recognition")
    print("=" * 60)
    
    # Initialize recognizer
    recognizer = BrandRecognizer(
        use_spacy=False, 
        fuzzy_threshold=65,
        exact_threshold=90
    )
    
    print(f"✅ Initialized with {len(recognizer.brands)} brands")
    print(f"✅ Built {len(recognizer.search_terms)} search terms")
    print()
    
    # Test cases
    test_cases = [
        "I love my new iPhone from Apple",
        "Samsung Galaxy is great",
        "Tesla Model 3 electric car",
        "iPhone vs Galaxy comparison",
        "Appl iPhone typo test",  # Intentional typo
        "Samsng Galaxy typo test",  # Intentional typo
        "McDonalds has good food",
        "Google search engine"
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"Test {i}: {text}")
        print("-" * 40)
        
        # Detect brands
        brands = recognizer.recognize_brands(text, return_details=True)
        
        if brands:
            for brand in brands:
                conf = brand['confidence']
                match_type = brand['match_type']
                matched_text = brand['matched_text']
                matched_term = brand.get('matched_term', 'N/A')
                
                print(f"  ✓ {brand['brand']}")
                print(f"    Confidence: {conf:.3f}")
                print(f"    Match type: {match_type}")
                print(f"    Matched text: '{matched_text}'")
                print(f"    Matched term: '{matched_term}'")
                
                if 'scorer_details' in brand:
                    scores = brand['scorer_details']
                    print(f"    Fuzzy scores: ratio={scores['ratio']}, partial={scores['partial_ratio']}")
                print()
        else:
            print("  ❌ No brands detected")
            print()
    
    # Test typo resistance
    print("🎯 Typo Resistance Test")
    print("=" * 60)
    
    typo_pairs = [
        ("Apple iPhone", "Appl iPhone"),
        ("Samsung Galaxy", "Samsng Galaxy"),
        ("Tesla Model", "Tesle Model"),
        ("Google Pixel", "Gogle Pixel")
    ]
    
    for correct, typo in typo_pairs:
        correct_brands = recognizer.recognize_brands(correct, return_details=False)
        typo_brands = recognizer.recognize_brands(typo, return_details=False)
        
        correct_detected = len(correct_brands) > 0
        typo_detected = len(typo_brands) > 0
        
        status = "✅" if typo_detected else "❌"
        
        print(f"{status} '{correct}' -> {correct_brands}")
        print(f"    '{typo}' -> {typo_brands}")
        print()

if __name__ == "__main__":
    test_fuzzywuzzy_approach() 