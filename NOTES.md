# NOTES

My model predicts end-of-turn probability using only causal audio available before each pause_start.  
It extracts short-term energy statistics, energy slope and drop, and final energy relative to the preceding speech window.  
It also uses autocorrelation F0 features: final pitch, average pitch, pitch variation, and the slope of the final voiced frames.  
Additional cues are voicing ratio, duration of the last continuous voiced run, a speaking-rate proxy, and pause position in the turn.  
A regularized histogram gradient-boosting classifier combines these 16 features and outputs p_eot for every annotated pause.  
The English-trained model achieved 947 ms mean response delay at 4.0% interrupted turns on the unseen Hindi folder, compared with a 1600 ms silence-only baseline.  
The model can still fail when a speaker uses atypical intonation, noise corrupts pitch/energy estimation, or a continuation pause has prosody similar to sentence completion.  
With one more day, I would evaluate grouped cross-validation across turns, inspect the highest-confidence false cutoffs by listening, add more robust voiced-segment and spectral features, and calibrate probabilities on held-out turns.