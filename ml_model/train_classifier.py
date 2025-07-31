import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import logging
from typing import List, Tuple, Dict, Any
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmailClassifierTrainer:
    def __init__(self):
        self.pipeline = None
        self.categories = [
            "work", "personal", "spam", "finance", "shopping", 
            "travel", "health", "education", "social", "other"
        ]
        
    def generate_training_data(self) -> Tuple[List[str], List[str]]:
        """Generate comprehensive training data with synthetic examples."""
        logger.info("Generating training data...")
        
        training_data = []
        
        # Work emails
        work_examples = [
            ("meeting tomorrow at 10am", "work"),
            ("application status update", "work"),
            ("project deadline extension", "work"),
            ("quarterly report review", "work"),
            ("team collaboration update", "work"),
            ("client presentation feedback", "work"),
            ("budget approval needed", "work"),
            ("new hire onboarding", "work"),
            ("performance review scheduled", "work"),
            ("conference call details", "work"),
            ("project milestone achieved", "work"),
            ("urgent action required", "work"),
            ("stakeholder meeting", "work"),
            ("code review request", "work"),
            ("deployment schedule", "work"),
            ("bug fix priority", "work"),
        ]
        
        # Personal emails
        personal_examples = [
            ("family dinner this weekend", "personal"),
            ("birthday party invitation", "personal"),
            ("vacation photos shared", "personal"),
            ("friend's wedding details", "personal"),
            ("family health update", "personal"),
            ("dinner plans tonight", "personal"),
            ("weekend trip planning", "personal"),
            ("anniversary celebration", "personal"),
            ("family reunion details", "personal"),
            ("friend's graduation", "personal"),
            ("holiday plans", "personal"),
            ("family movie night", "personal"),
            ("friend's new job", "personal"),
            ("birthday wishes", "personal"),
            ("family vacation photos", "personal"),
        ]
        
        # Finance emails
        finance_examples = [
            ("bank statement available", "finance"),
            ("credit card payment due", "finance"),
            ("investment portfolio update", "finance"),
            ("tax filing reminder", "finance"),
            ("insurance policy renewal", "finance"),
            ("mortgage payment confirmation", "finance"),
            ("investment opportunity", "finance"),
            ("retirement account update", "finance"),
            ("loan application status", "finance"),
            ("credit score update", "finance"),
            ("budget review meeting", "finance"),
            ("expense report approval", "finance"),
            ("financial planning session", "finance"),
            ("investment advice", "finance"),
            ("tax deduction reminder", "finance"),
        ]
        
        # Shopping emails
        shopping_examples = [
            ("order confirmation", "shopping"),
            ("sale promotion", "shopping"),
            ("delivery tracking", "shopping"),
            ("return policy", "shopping"),
            ("product review request", "shopping"),
            ("flash sale alert", "shopping"),
            ("order shipped", "shopping"),
            ("discount code", "shopping"),
            ("wishlist item on sale", "shopping"),
            ("product recommendation", "shopping"),
            ("order cancellation", "shopping"),
            ("refund processed", "shopping"),
            ("loyalty points earned", "shopping"),
            ("new product launch", "shopping"),
            ("customer satisfaction survey", "shopping"),
        ]
        
        # Spam emails
        spam_examples = [
            ("win free money", "spam"),
            ("urgent action required", "spam"),
            ("limited time offer", "spam"),
            ("claim your prize", "spam"),
            ("exclusive deal", "spam"),
            ("you've been selected", "spam"),
            ("act now or miss out", "spam"),
            ("guaranteed income", "spam"),
            ("work from home", "spam"),
            ("lose weight fast", "spam"),
            ("investment opportunity", "spam"),
            ("inheritance claim", "spam"),
            ("lottery winner", "spam"),
            ("pharmaceutical offer", "spam"),
            ("dating site invitation", "spam"),
        ]
        
        # Travel emails
        travel_examples = [
            ("flight booking confirmation", "travel"),
            ("hotel reservation", "travel"),
            ("car rental details", "travel"),
            ("travel insurance", "travel"),
            ("vacation package", "travel"),
            ("airport transfer", "travel"),
            ("tour booking", "travel"),
            ("travel itinerary", "travel"),
            ("flight delay notification", "travel"),
            ("hotel check-in", "travel"),
            ("travel visa application", "travel"),
            ("cruise booking", "travel"),
            ("travel rewards", "travel"),
            ("destination guide", "travel"),
            ("travel advisory", "travel"),
        ]
        
        # Health emails
        health_examples = [
            ("doctor appointment", "health"),
            ("medical test results", "health"),
            ("prescription refill", "health"),
            ("health insurance", "health"),
            ("dental appointment", "health"),
            ("pharmacy notification", "health"),
            ("health screening", "health"),
            ("medical bill", "health"),
            ("vaccination reminder", "health"),
            ("health checkup", "health"),
            ("medication reminder", "health"),
            ("healthcare provider", "health"),
            ("medical consultation", "health"),
            ("health benefits", "health"),
            ("wellness program", "health"),
        ]
        
        # Education emails
        education_examples = [
            ("course registration", "education"),
            ("assignment due", "education"),
            ("exam schedule", "education"),
            ("academic calendar", "education"),
            ("student loan", "education"),
            ("scholarship application", "education"),
            ("library notification", "education"),
            ("academic advisor", "education"),
            ("course materials", "education"),
            ("graduation ceremony", "education"),
            ("research project", "education"),
            ("academic transcript", "education"),
            ("study abroad", "education"),
            ("online course", "education"),
            ("academic conference", "education"),
        ]
        
        # Social emails
        social_examples = [
            ("social media notification", "social"),
            ("friend request", "social"),
            ("event invitation", "social"),
            ("group message", "social"),
            ("social network", "social"),
            ("community event", "social"),
            ("social gathering", "social"),
            ("online community", "social"),
            ("social platform", "social"),
            ("social networking", "social"),
            ("social event", "social"),
            ("social group", "social"),
            ("social media", "social"),
            ("social club", "social"),
            ("social activity", "social"),
        ]
        
        # Combine all examples
        all_examples = (
            work_examples + personal_examples + finance_examples + 
            shopping_examples + spam_examples + travel_examples + 
            health_examples + education_examples + social_examples
        )
        
        # Add some "other" category examples
        other_examples = [
            ("newsletter subscription", "other"),
            ("system notification", "other"),
            ("account verification", "other"),
            ("password reset", "other"),
            ("service update", "other"),
            ("maintenance notification", "other"),
            ("policy update", "other"),
            ("terms of service", "other"),
            ("privacy policy", "other"),
            ("technical support", "other"),
        ]
        
        all_examples.extend(other_examples)
        
        # Shuffle the data
        np.random.shuffle(all_examples)
        
        texts, labels = zip(*all_examples)
        return list(texts), list(labels)
    
    def create_pipeline(self) -> Pipeline:
        """Create a machine learning pipeline with TF-IDF and Random Forest."""
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=2000,
                stop_words='english',
                ngram_range=(1, 3),
                min_df=2,
                max_df=0.95
            )),
            ('classifier', RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ))
        ])
        return pipeline
    
    def train_model(self, save_model: bool = True) -> Dict[str, Any]:
        """Train the email classifier and return performance metrics."""
        logger.info("Starting model training...")
        
        # Generate training data
        texts, labels = self.generate_training_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Create and train pipeline
        self.pipeline = self.create_pipeline()
        
        logger.info("Training pipeline...")
        self.pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.pipeline.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        
        # Cross-validation
        cv_scores = cross_val_score(self.pipeline, texts, labels, cv=5)
        
        # Save model if requested
        if save_model:
            self.save_model()
        
        # Prepare results
        results = {
            "accuracy": accuracy,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "classification_report": classification_rep,
            "test_samples": len(X_test),
            "train_samples": len(X_train),
            "total_samples": len(texts),
            "categories": self.categories
        }
        
        logger.info(f"Training completed. Accuracy: {accuracy:.3f}")
        logger.info(f"Cross-validation score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return results
    
    def save_model(self, model_path: str = "ml_model/email_classifier.pkl"):
        """Save the trained model."""
        os.makedirs("ml_model", exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.pipeline, f)
        
        # Save vectorizer separately for compatibility
        vectorizer_path = "ml_model/vectorizer.pkl"
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.pipeline.named_steps['tfidf'], f)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Vectorizer saved to {vectorizer_path}")
    
    def load_model(self, model_path: str = "ml_model/email_classifier.pkl"):
        """Load a trained model."""
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.pipeline = pickle.load(f)
            logger.info(f"Model loaded from {model_path}")
            return True
        else:
            logger.warning(f"Model file not found: {model_path}")
            return False
    
    def predict(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Predict categories for given texts."""
        if self.pipeline is None:
            raise ValueError("Model not trained or loaded")
        
        predictions = self.pipeline.predict(texts)
        probabilities = self.pipeline.predict_proba(texts)
        
        results = []
        for pred, prob in zip(predictions, probabilities):
            confidence = max(prob)
            results.append((pred, confidence))
        
        return results
    
    def evaluate_model(self, test_texts: List[str], test_labels: List[str]) -> Dict[str, Any]:
        """Evaluate model performance on test data."""
        if self.pipeline is None:
            raise ValueError("Model not trained or loaded")
        
        predictions = self.pipeline.predict(test_texts)
        probabilities = self.pipeline.predict_proba(test_texts)
        
        accuracy = accuracy_score(test_labels, predictions)
        classification_rep = classification_report(test_labels, predictions, output_dict=True)
        
        # Calculate confidence scores
        confidence_scores = [max(prob) for prob in probabilities]
        avg_confidence = np.mean(confidence_scores)
        
        return {
            "accuracy": accuracy,
            "classification_report": classification_rep,
            "average_confidence": avg_confidence,
            "predictions": list(zip(predictions, confidence_scores))
        }

def main():
    """Main training function."""
    trainer = EmailClassifierTrainer()
    
    # Train the model
    results = trainer.train_model(save_model=True)
    
    # Print results
    print("\n" + "="*50)
    print("EMAIL CLASSIFIER TRAINING RESULTS")
    print("="*50)
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(f"Cross-validation: {results['cv_mean']:.3f} (+/- {results['cv_std'] * 2:.3f})")
    print(f"Total samples: {results['total_samples']}")
    print(f"Test samples: {results['test_samples']}")
    print(f"Categories: {', '.join(results['categories'])}")
    
    # Save results to file
    with open("ml_model/training_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nTraining completed successfully!")
    print("Model saved to ml_model/email_classifier.pkl")
    print("Results saved to ml_model/training_results.json")

if __name__ == "__main__":
    main()
