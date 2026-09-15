import nltk
from nltk.corpus import udhr
from nltk.collocations import TrigramAssocMeasures, TrigramCollocationFinder
import speech_recognition as sr
from wordfreq import zipf_frequency
from pathlib import Path

class SLID:
    def __init__(self):
        self.langs = {
            "en": "en-US",
            "nl": "nl-NL",
            "fr": "fr-FR",
            "it": "it-IT",
            "es": "es-ES",
            "de" : "de-DE"
            }
    
        self.language_files = {
                        "en": "English-Latin1",
                        "fr": "French_Francais-Latin1",
                        "it": "Italian_Italiano-Latin1",
                        "es": "Spanish_Espanol-Latin1",
                        "nl": "Dutch_Nederlands-Latin1",
                        "de" : "German_Deutsch-Latin1"
                        }

        self.training_texts = {language : udhr.raw(file_id) for (language, file_id) in self.language_files.items()}
        self.predictions_scores = {}

    def _normalize(self, text):
        tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
        return tokenizer.tokenize(text.lower())

    def _evaluate_transcript(self, words, language):
        if not words:
            return 0.0

        valid_words = 0
        for word in words:
            if zipf_frequency(word, language) >= 2:
                valid_words += 1
            
        valid_ratio = valid_words / len(words)
        return valid_ratio

    def _trigrams(self, words):
        trigram_measures = TrigramAssocMeasures()
        finder = TrigramCollocationFinder.from_words(words)
        scored = dict(finder.score_ngrams(trigram_measures.raw_freq))
        return scored

    def predict(self, audio_path, return_scores = False):
        training_profiles = {language : self._trigrams(list(" ".join(self._normalize(text)))) for language, text in self.training_texts.items()}
        rec = sr.Recognizer()

        with sr.AudioFile(str(audio_path)) as source:
            audio = rec.record(source)
            results = {l : {} for l in self.langs}

            for lang in self.langs:
                try:
                    predictions = rec.recognize_google(audio, language = self.langs[lang], show_all = True)
                    alternatives = (predictions.get("alternative", []) if isinstance(predictions, dict) else [])
                    best_result = alternatives[0]
                    best_trans = best_result.get("transcript", "")
                    best_confidence = best_result.get("confidence", 0)
                    words = self._normalize(best_trans)

                    trigram = self._trigrams(list(" ".join(words)))

                    total_trigrams = len(trigram)
                    occured_trigrams = 0

                    if trigram:
                        for tri in trigram:
                            if tri in training_profiles[lang]:
                                occured_trigrams += 1
                        
                        trigram_coverage = occured_trigrams / total_trigrams

                    else:
                        trigram_coverage = 0.0

                    results[lang] = {"transcription" : best_trans, 
                                    "words" : words,
                                    "words_length" : len(words),
                                    "confidence" : best_confidence,
                                    "trigram_coverage" : trigram_coverage,
                                    "evaluation" : self._evaluate_transcript(self._normalize(best_trans), lang)}
                except:
                    results[lang] = {"transcription" : "", 
                                    "words" : [],
                                    "words_length" : 0.0,
                                    "confidence" : 0.0,
                                    "trigram_coverage" : 0.0,
                                    "evaluation" : 0.0}

            maximum_word_length = max(result["words_length"] for result in results.values())
            
            for lang in self.langs:
                if maximum_word_length:
                    length_ratio = results[lang]["words_length"] / maximum_word_length
                else:
                    length_ratio = 0.0
                
                base_score = (
                    0.40 * results[lang]["confidence"]
                    + 0.35 * results[lang]["trigram_coverage"]
                    + 0.25 * results[lang]["evaluation"]
                )

                final_score = base_score * (
                    0.25 + 0.75 * length_ratio
                )

                self.predictions_scores[lang] = final_score

        answer = max(self.predictions_scores, key = self.predictions_scores.get)

        if return_scores:
            return (answer, self.predictions_scores)
        return answer