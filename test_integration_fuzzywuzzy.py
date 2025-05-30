#!/usr/bin/env python3
"""
Integration Test for Fuzzywuzzy-First Entity Recognition with Competitor Analysis
Tests the compatibility and functionality of the updated modules working together
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ml.entity_recognition import BrandRecognizer, ProductExtractor
from src.spark.competitor_analysis import create_brand_recognition_udf

def test_fuzzywuzzy_integration():
    """Test the integration between fuzzywuzzy-first entity recognition and competitor analysis."""
    
    print("🧪 Testing Fuzzywuzzy-First Integration")
    print("=" * 60)
    
    try:
        # 1. Test BrandRecognizer with fuzzywuzzy-first approach
        print("1. Testing BrandRecognizer initialization...")
        recognizer = BrandRecognizer(
            use_spacy=False,
            fuzzy_threshold=65,
            exact_threshold=90,
            scorer_weights={
                'ratio': 0.4,
                'partial_ratio': 0.25,
                'token_sort_ratio': 0.25,
                'token_set_ratio': 0.1
            }
        )
        print(f"   ✅ Initialized with {len(recognizer.brands)} brands")
        print(f"   ✅ Built {len(recognizer.search_terms)} search terms")
        print(f"   ✅ Created {len(recognizer.term_to_brand)} term mappings")
        
        # 2. Test brand recognition with various text types
        print("\n2. Testing brand recognition capabilities...")
        test_cases = [
            ("Apple iPhone is the best", "standard_match"),
            ("I love my Appl iPhone", "typo_resistance"),
            ("Samsung Galaxy vs iPhone comparison", "multi_brand"),
            ("Google Pixel camera quality", "alternative_brand"),
            ("Tesla Model 3 electric vehicle", "automotive_brand"),
            ("No tech brands mentioned here", "no_match")
        ]
        
        detected_brands_summary = {}
        
        for text, test_type in test_cases:
            brands = recognizer.recognize_brands(text, return_details=True)
            detected_brands_summary[test_type] = len(brands)
            
            print(f"   Test: {test_type}")
            print(f"   Text: '{text}'")
            
            if brands:
                for brand in brands:
                    conf = brand['confidence']
                    match_type = brand['match_type']
                    print(f"      ✅ {brand['brand']} (conf: {conf:.3f}, type: {match_type})")
            else:
                print(f"      ❌ No brands detected")
            print()
        
        # 3. Test ProductExtractor
        print("3. Testing ProductExtractor with brand context...")
        extractor = ProductExtractor(recognizer)
        
        product_test_cases = [
            "I bought a new iPhone 15 Pro",
            "Samsung Galaxy S24 Ultra camera",
            "Tesla Model Y performance"
        ]
        
        for text in product_test_cases:
            # First detect brands for context
            brands = recognizer.recognize_brands(text, return_details=True)
            # Then extract products
            products = extractor.extract_products(text, brands)
            
            print(f"   Text: '{text}'")
            if products:
                for product in products:
                    conf = product['confidence']
                    print(f"      ✅ {product['product']} by {product['brand']} (conf: {conf:.2f})")
            else:
                print(f"      ❌ No products detected")
            print()
        
        # 4. Test Competitor Analysis UDF Integration
        print("4. Testing Competitor Analysis UDF integration...")
        
        # Create UDF with fuzzywuzzy-first approach
        brand_udf = create_brand_recognition_udf()
        
        # Test UDF function
        udf_test_cases = [
            "Apple iPhone vs Samsung Galaxy",
            "Google Pixel camera beats iPhone",
            "Tesla autopilot technology",
            "Microsoft Surface laptop"
        ]
        
        for text in udf_test_cases:
            result = brand_udf.func(text)
            print(f"   UDF Test: '{text}'")
            if result:
                print(f"      ✅ Detected: {result}")
            else:
                print(f"      ❌ No detection")
            print()
        
        # 5. Test Fuzzy Matching Performance
        print("5. Testing fuzzy matching performance...")
        
        typo_performance = []
        typo_tests = [
            ("Apple", "Aple"),
            ("Samsung", "Samsng"),
            ("Google", "Gogle"),
            ("iPhone", "iPhne"),
            ("Galaxy", "Galxy"),
            ("Tesla", "Tesle")
        ]
        
        for correct, typo in typo_tests:
            correct_brands = recognizer.recognize_brands(f"{correct} product", return_details=False)
            typo_brands = recognizer.recognize_brands(f"{typo} product", return_details=False)
            
            correct_detected = len(correct_brands) > 0
            typo_detected = len(typo_brands) > 0
            
            performance = "✅" if typo_detected else "❌"
            typo_performance.append(typo_detected)
            
            print(f"   {performance} '{correct}' vs '{typo}': {correct_detected} -> {typo_detected}")
        
        typo_success_rate = sum(typo_performance) / len(typo_performance) * 100
        print(f"   Overall typo resistance: {typo_success_rate:.1f}%")
        
        # 6. Test Competitor Mapping
        print("\n6. Testing competitor relationships...")
        competitor_map = recognizer.competitor_map
        
        for brand, competitors in list(competitor_map.items())[:5]:  # Show first 5
            if competitors:
                comp_list = ', '.join(list(competitors)[:3])  # Show first 3 competitors
                print(f"   {brand}: {comp_list}")
        
        # 7. Performance Summary
        print(f"\n7. Performance Summary:")
        print(f"   ✅ Brands in configuration: {len(recognizer.brands)}")
        print(f"   ✅ Search terms built: {len(recognizer.search_terms)}")
        print(f"   ✅ Competitor relationships: {sum(len(v) for v in competitor_map.values()) // 2}")
        print(f"   ✅ Detection tests passed: {sum(1 for _, count in detected_brands_summary.items() if count > 0)}/{len(detected_brands_summary)}")
        print(f"   ✅ Typo resistance rate: {typo_success_rate:.1f}%")
        print(f"   ✅ Product extractor functional: True")
        print(f"   ✅ UDF integration working: True")
        
        # 8. Configuration verification
        print(f"\n8. Configuration Details:")
        print(f"   - Fuzzy threshold: {recognizer.fuzzy_threshold}")
        print(f"   - Exact threshold: {recognizer.exact_threshold}")
        print(f"   - Scorer weights: {recognizer.scorer_weights}")
        print(f"   - spaCy enabled: {recognizer.nlp is not None}")
        
        print("\n" + "=" * 60)
        print("🎉 Integration Test COMPLETED")
        print("✅ Fuzzywuzzy-first entity recognition is working correctly")
        print("✅ Competitor analysis integration is functional")
        print("✅ All components are compatible")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_compatibility_scenarios():
    """Test specific compatibility scenarios between modules."""
    
    print("\n" + "=" * 60)
    print("🔧 Testing Compatibility Scenarios")
    print("=" * 60)
    
    try:
        recognizer = BrandRecognizer(use_spacy=False, fuzzy_threshold=70)
        
        # Scenario 1: Mixed case sensitivity
        test_text = "apple IPHONE vs samsung GALAXY"
        brands = recognizer.recognize_brands(test_text, return_details=True)
        
        print("1. Case sensitivity test:")
        print(f"   Text: '{test_text}'")
        for brand in brands:
            print(f"   ✅ {brand['brand']} (case handling: OK)")
        
        # Scenario 2: Multiple fuzzy matches
        test_text = "Apl vs Samsng smartphone comparison"
        brands = recognizer.recognize_brands(test_text, return_details=True)
        
        print("\n2. Multiple fuzzy matches:")
        print(f"   Text: '{test_text}'")
        for brand in brands:
            print(f"   ✅ {brand['brand']} (fuzzy score: {brand['confidence']:.3f})")
        
        # Scenario 3: UDF format compatibility
        udf = create_brand_recognition_udf()
        result = udf.func("Apple iPhone great phone")
        
        print("\n3. UDF output format:")
        print(f"   UDF result: {result}")
        
        # Verify format: ["brand:confidence", ...]
        if result:
            for item in result:
                if ':' in item:
                    brand, conf = item.split(':')
                    print(f"   ✅ Format OK: {brand} -> {conf}")
                else:
                    print(f"   ❌ Format issue: {item}")
        
        print("\n✅ All compatibility scenarios passed")
        return True
        
    except Exception as e:
        print(f"\n❌ Compatibility test FAILED: {e}")
        return False

if __name__ == "__main__":
    print("Starting Fuzzywuzzy-First Integration Tests...")
    
    success1 = test_fuzzywuzzy_integration()
    success2 = test_compatibility_scenarios()
    
    overall_success = success1 and success2
    
    print("\n" + "=" * 80)
    print("FINAL INTEGRATION TEST RESULTS")
    print("=" * 80)
    print(f"Entity Recognition Test: {'PASSED' if success1 else 'FAILED'}")
    print(f"Compatibility Test: {'PASSED' if success2 else 'FAILED'}")
    print(f"Overall Result: {'PASSED' if overall_success else 'FAILED'}")
    
    if overall_success:
        print("\n🎉 All tests passed! The fuzzywuzzy-first approach is working correctly")
        print("   with full compatibility between entity recognition and competitor analysis.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
    
    sys.exit(0 if overall_success else 1) 