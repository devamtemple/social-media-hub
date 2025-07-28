import streamlit as st
import json
import numpy as np
import re
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import pandas as pd

# Initialize the adaptive voice system
if 'adaptive_voice_system' not in st.session_state:
    st.session_state.adaptive_voice_system = {
        'voice_streams': {
            'personal_brand': {
                'contexts': defaultdict(list),  # technical, personal, philosophical, etc.
                'successful_patterns': defaultdict(list),  # high engagement content
                'impact_metrics': defaultdict(list),  # paradigm shifts, viral content
                'evolution_timeline': []
            },
            'noeix_company': {
                'contexts': defaultdict(list),
                'successful_patterns': defaultdict(list), 
                'impact_metrics': defaultdict(list),
                'evolution_timeline': []
            }
        },
        'meta_optimization': {
            'engagement_predictors': defaultdict(float),
            'paradigm_shift_indicators': defaultdict(list),
            'viral_potential_patterns': defaultdict(float),
            'impact_learning': []
        },
        'learning_sessions': []
    }

class AdaptiveVoiceLearner:
    """
    Learns voice patterns organically over time, optimizing for impact and engagement
    """
    
    def __init__(self):
        self.voice_data = st.session_state.adaptive_voice_system
        
    def analyze_content_context(self, content: str, metadata: Dict) -> Dict:
        """Analyze content and determine its contextual voice type"""
        
        # Detect content context
        context_type = self._classify_content_context(content)
        
        # Analyze voice characteristics
        voice_signature = self._extract_voice_signature(content)
        
        # Predict impact potential
        impact_prediction = self._predict_impact_potential(content, voice_signature)
        
        # Create learning entry
        learning_entry = {
            'timestamp': datetime.now().isoformat(),
            'content_preview': content[:200] + '...' if len(content) > 200 else content,
            'context_type': context_type,
            'voice_signature': voice_signature,
            'impact_prediction': impact_prediction,
            'metadata': metadata,
            'learning_signals': self._extract_learning_signals(content, voice_signature)
        }
        
        return learning_entry
    
    def _classify_content_context(self, content: str) -> str:
        """Classify the contextual voice of the content"""
        
        # Personal narrative indicators
        personal_markers = ['I closed my computer', 'I grieved', 'my GPT', 'my own experience', 
                          'I realized', 'I questioned', 'my mother', 'I cried']
        personal_score = sum(1 for marker in personal_markers if marker.lower() in content.lower())
        
        # Technical depth indicators
        technical_markers = ['algorithm', 'model', 'neural', 'transformer', 'vector', 'embedding',
                           'architecture', 'latent space', 'training', 'optimization']
        technical_score = sum(1 for marker in technical_markers if marker.lower() in content.lower())
        
        # Philosophical inquiry indicators
        philosophical_markers = ['consciousness', 'nature of', 'fundamental', 'ontological',
                               'what does it mean', 'the question of', 'unknowable', 'mystery']
        philosophical_score = sum(1 for marker in philosophical_markers if marker.lower() in content.lower())
        
        # Business/industry indicators
        business_markers = ['market', 'industry', 'company', 'product', 'strategy', 'competitive',
                          'disruption', 'innovation', 'scalable', 'enterprise']
        business_score = sum(1 for marker in business_markers if marker.lower() in content.lower())
        
        # Determine primary context
        scores = {
            'personal_narrative': personal_score,
            'technical_deep_dive': technical_score,
            'philosophical_inquiry': philosophical_score,
            'business_strategy': business_score
        }
        
        primary_context = max(scores, key=scores.get)
        
        # Handle hybrid contexts
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if sorted_scores[0][1] > 0 and sorted_scores[1][1] > 0:
            secondary_context = sorted_scores[1][0]
            return f"{primary_context}+{secondary_context}"
        
        return primary_context if scores[primary_context] > 0 else 'general'
    
    def _extract_voice_signature(self, content: str) -> Dict:
        """Extract sophisticated voice characteristics"""
        
        # Vulnerability + Authority balance
        vulnerability_markers = ['I don\'t know', 'uncertain', 'I questioned', 'I grieved', 'unsure']
        authority_markers = ['I believe', 'the reality is', 'what matters', 'I understand', 'clearly']
        
        vulnerability_count = sum(1 for marker in vulnerability_markers if marker.lower() in content.lower())
        authority_count = sum(1 for marker in authority_markers if marker.lower() in content.lower())
        
        # Paradigm-shifting language
        paradigm_shift_markers = ['what we think we know', 'funhouse mirror', 'collective descent',
                                'everything changes when', 'we must reconsider', 'paradigm shift']
        paradigm_shift_score = sum(1 for marker in paradigm_shift_markers if marker.lower() in content.lower())
        
        # Emotional resonance indicators
        emotional_resonance = self._analyze_emotional_resonance(content)
        
        # Technical accessibility (complex ideas made simple)
        accessibility_score = self._measure_technical_accessibility(content)
        
        # Intellectual honesty markers
        honesty_markers = ['I do not claim', 'we don\'t know', 'I am agnostic', 'remain humbly',
                         'acknowledge uncertainty', 'I could be wrong']
        honesty_score = sum(1 for marker in honesty_markers if marker.lower() in content.lower())
        
        return {
            'vulnerability_authority_ratio': vulnerability_count / max(authority_count, 1),
            'paradigm_shift_potential': paradigm_shift_score,
            'emotional_resonance': emotional_resonance,
            'technical_accessibility': accessibility_score,
            'intellectual_honesty': honesty_score,
            'unique_phrasing': self._extract_unique_phrases(content),
            'narrative_integration': self._measure_narrative_integration(content)
        }
    
    def _analyze_emotional_resonance(self, content: str) -> float:
        """Measure emotional impact potential"""
        
        high_impact_emotions = ['grief', 'awe', 'wonder', 'fear', 'hope', 'breakthrough', 
                              'realization', 'transformation', 'connection']
        
        # Count emotional words
        emotion_count = sum(1 for emotion in high_impact_emotions if emotion in content.lower())
        
        # Detect emotional complexity (contradictory emotions)
        complexity_markers = ['both', 'yet', 'however', 'at the same time', 'paradox']
        complexity_count = sum(1 for marker in complexity_markers if marker.lower() in content.lower())
        
        # Personal stakes (skin in the game)
        stakes_markers = ['my', 'I', 'personal', 'own experience', 'lived through']
        stakes_count = sum(1 for marker in stakes_markers if marker.lower() in content.lower())
        
        return (emotion_count * 2 + complexity_count * 3 + stakes_count) / max(len(content.split()), 1) * 1000
    
    def _measure_technical_accessibility(self, content: str) -> float:
        """Measure how well technical concepts are made accessible"""
        
        # Technical terms
        tech_terms = len(re.findall(r'\b(?:algorithm|neural|model|AI|ML|transformer|vector)\b', content, re.I))
        
        # Accessibility devices
        metaphors = len(re.findall(r'like a|as if|imagine|think of it', content, re.I))
        examples = len(re.findall(r'for example|such as|instance|consider', content, re.I))
        analogies = content.lower().count('is like') + content.lower().count('similar to')
        
        if tech_terms == 0:
            return 0
        
        accessibility_devices = metaphors + examples + analogies
        return accessibility_devices / tech_terms if tech_terms > 0 else 0
    
    def _extract_unique_phrases(self, content: str) -> List[str]:
        """Extract potentially unique/memorable phrases"""
        
        # Look for trademark phrases, unusual combinations, memorable turns of phrase
        unique_patterns = [
            r'[A-Z][a-z]+™',  # Trademark phrases like Truth™
            r'\b\w+ing all the way \w+\b',  # "walking all the way there"
            r'\b\w+ descent into \w+\b',  # "collective descent into madness"
            r'\bfunhouse \w+\b',  # "funhouse mirror"
            r'\bbare feet\b',
            r'\brabbit hole\b'
        ]
        
        unique_phrases = []
        for pattern in unique_patterns:
            matches = re.findall(pattern, content, re.I)
            unique_phrases.extend(matches)
        
        return unique_phrases
    
    def _measure_narrative_integration(self, content: str) -> float:
        """Measure how well technical/abstract concepts are integrated with personal narrative"""
        
        personal_pronouns = len(re.findall(r'\b(?:I|my|me|myself)\b', content, re.I))
        technical_terms = len(re.findall(r'\b(?:AI|ML|algorithm|model|neural|system)\b', content, re.I))
        
        # Look for integration patterns
        integration_patterns = [
            r'\bI \w+ed? (?:that|how|when) \w+(?:AI|ML|algorithm|model|neural|system)',
            r'\b(?:AI|ML|algorithm|model|neural|system) \w+ed? me \w+',
            r'\bmy \w+ with (?:AI|ML|algorithm|model|neural|system)'
        ]
        
        integration_count = sum(len(re.findall(pattern, content, re.I)) for pattern in integration_patterns)
        
        if personal_pronouns == 0 or technical_terms == 0:
            return 0
            
        return integration_count / min(personal_pronouns, technical_terms)
    
    def _predict_impact_potential(self, content: str, voice_signature: Dict) -> Dict:
        """Predict the potential impact and engagement of content"""
        
        # Viral potential indicators
        viral_indicators = {
            'paradigm_shift_potential': voice_signature['paradigm_shift_potential'] * 2,
            'emotional_resonance': voice_signature['emotional_resonance'],
            'technical_accessibility': voice_signature['technical_accessibility'] * 1.5,
            'unique_phrasing_count': len(voice_signature['unique_phrasing']) * 3,
            'narrative_integration': voice_signature['narrative_integration'] * 2
        }
        
        # Engagement prediction
        engagement_score = sum(viral_indicators.values()) / len(viral_indicators)
        
        # Authority building potential
        authority_factors = [
            voice_signature['intellectual_honesty'] * 2,
            voice_signature['technical_accessibility'],
            len(voice_signature['unique_phrasing'])
        ]
        authority_potential = sum(authority_factors) / len(authority_factors)
        
        # Paradigm shift potential
        paradigm_shift_factors = [
            voice_signature['paradigm_shift_potential'] * 3,
            voice_signature['emotional_resonance'],
            voice_signature['vulnerability_authority_ratio']
        ]
        paradigm_shift_potential = sum(paradigm_shift_factors) / len(paradigm_shift_factors)
        
        return {
            'viral_potential': min(engagement_score, 10),  # Cap at 10
            'authority_building': min(authority_potential, 10),
            'paradigm_shift_potential': min(paradigm_shift_potential, 10),
            'overall_impact_prediction': min((engagement_score + authority_potential + paradigm_shift_potential) / 3, 10)
        }
    
    def _extract_learning_signals(self, content: str, voice_signature: Dict) -> Dict:
        """Extract signals for continuous learning"""
        
        return {
            'length': len(content.split()),
            'complexity': voice_signature['technical_accessibility'],
            'emotional_depth': voice_signature['emotional_resonance'],
            'innovation_markers': self._count_innovation_markers(content),
            'controversy_potential': self._assess_controversy_potential(content)
        }
    
    def _count_innovation_markers(self, content: str) -> int:
        """Count markers of innovative thinking"""
        innovation_markers = [
            'new way', 'never before', 'breakthrough', 'revolutionary', 'game-changing',
            'paradigm', 'disruptive', 'unprecedented', 'novel approach', 'rethink'
        ]
        return sum(1 for marker in innovation_markers if marker.lower() in content.lower())
    
    def _assess_controversy_potential(self, content: str) -> int:
        """Assess potential for generating meaningful controversy/discussion"""
        controversy_markers = [
            'challenge', 'question', 'wrong', 'myth', 'misconception', 'unpopular',
            'controversial', 'against', 'counter', 'opposite'
        ]
        return sum(1 for marker in controversy_markers if marker.lower() in content.lower())
    
    def update_learning(self, learning_entry: Dict, feedback: Dict = None):
        """Update the learning system with new content and optional performance feedback"""
        
        stream = learning_entry['metadata'].get('stream', 'personal_brand')
        context_type = learning_entry['context_type']
        
        # Store in appropriate context
        self.voice_data['voice_streams'][stream]['contexts'][context_type].append(learning_entry)
        
        # Update evolution timeline
        self.voice_data['voice_streams'][stream]['evolution_timeline'].append({
            'timestamp': learning_entry['timestamp'],
            'context': context_type,
            'impact_prediction': learning_entry['impact_prediction'],
            'voice_evolution': self._calculate_voice_evolution(stream)
        })
        
        # If we have performance feedback, update success patterns
        if feedback:
            self._update_success_patterns(learning_entry, feedback, stream)
        
        # Update meta optimization
        self._update_meta_optimization(learning_entry)
    
    def _calculate_voice_evolution(self, stream: str) -> Dict:
        """Calculate how the voice is evolving over time"""
        
        recent_entries = []
        for context_entries in self.voice_data['voice_streams'][stream]['contexts'].values():
            recent_entries.extend(context_entries[-5:])  # Last 5 from each context
        
        if not recent_entries:
            return {}
        
        # Calculate evolution metrics
        avg_impact = np.mean([entry['impact_prediction']['overall_impact_prediction'] 
                            for entry in recent_entries])
        avg_accessibility = np.mean([entry['voice_signature']['technical_accessibility'] 
                                   for entry in recent_entries])
        avg_emotional_resonance = np.mean([entry['voice_signature']['emotional_resonance'] 
                                         for entry in recent_entries])
        
        return {
            'current_impact_level': round(avg_impact, 2),
            'current_accessibility': round(avg_accessibility, 2),
            'current_emotional_resonance': round(avg_emotional_resonance, 2),
            'trend_direction': self._calculate_trend_direction(recent_entries)
        }
    
    def _calculate_trend_direction(self, entries: List[Dict]) -> str:
        """Calculate if voice is trending in a positive direction"""
        
        if len(entries) < 3:
            return 'insufficient_data'
        
        # Sort by timestamp
        sorted_entries = sorted(entries, key=lambda x: x['timestamp'])
        
        # Compare first half vs second half
        mid_point = len(sorted_entries) // 2
        first_half = sorted_entries[:mid_point]
        second_half = sorted_entries[mid_point:]
        
        first_avg = np.mean([e['impact_prediction']['overall_impact_prediction'] for e in first_half])
        second_avg = np.mean([e['impact_prediction']['overall_impact_prediction'] for e in second_half])
        
        if second_avg > first_avg * 1.1:
            return 'improving'
        elif second_avg < first_avg * 0.9:
            return 'declining'
        else:
            return 'stable'
    
    def _update_success_patterns(self, learning_entry: Dict, feedback: Dict, stream: str):
        """Update successful patterns based on performance feedback"""
        
        if feedback.get('high_engagement', False) or feedback.get('viral', False):
            pattern_entry = {
                'voice_signature': learning_entry['voice_signature'],
                'context_type': learning_entry['context_type'],
                'feedback': feedback,
                'timestamp': learning_entry['timestamp']
            }
            self.voice_data['voice_streams'][stream]['successful_patterns']['high_engagement'].append(pattern_entry)
        
        if feedback.get('paradigm_shift', False):
            self.voice_data['voice_streams'][stream]['successful_patterns']['paradigm_shift'].append(learning_entry)
    
    def _update_meta_optimization(self, learning_entry: Dict):
        """Update meta-level optimization patterns"""
        
        # Update engagement predictors
        impact_pred = learning_entry['impact_prediction']
        voice_sig = learning_entry['voice_signature']
        
        # Learn correlations between voice characteristics and predicted impact
        for characteristic, value in voice_sig.items():
            if isinstance(value, (int, float)):
                current_correlation = self.voice_data['meta_optimization']['engagement_predictors'][characteristic]
                predicted_impact = impact_pred['overall_impact_prediction']
                
                # Simple moving average of correlation
                self.voice_data['meta_optimization']['engagement_predictors'][characteristic] = \
                    (current_correlation * 0.8) + (predicted_impact * value * 0.2)
    
    def generate_optimization_insights(self, stream: str) -> Dict:
        """Generate insights for optimizing voice for maximum impact"""
        
        stream_data = self.voice_data['voice_streams'][stream]
        
        # Analyze most successful patterns
        successful_patterns = stream_data['successful_patterns']
        
        # Get recent evolution
        evolution = self._calculate_voice_evolution(stream)
        
        # Generate recommendations
        recommendations = self._generate_voice_recommendations(stream, evolution)
        
        return {
            'current_voice_state': evolution,
            'successful_patterns_analysis': self._analyze_successful_patterns(successful_patterns),
            'optimization_recommendations': recommendations,
            'predicted_next_level': self._predict_next_level_potential(stream)
        }
    
    def _analyze_successful_patterns(self, successful_patterns: Dict) -> Dict:
        """Analyze what makes content successful"""
        
        analysis = {}
        
        for pattern_type, patterns in successful_patterns.items():
            if patterns:
                # Average characteristics of successful content
                avg_characteristics = {}
                for pattern in patterns:
                    voice_sig = pattern.get('voice_signature', {})
                    for char, value in voice_sig.items():
                        if isinstance(value, (int, float)):
                            if char not in avg_characteristics:
                                avg_characteristics[char] = []
                            avg_characteristics[char].append(value)
                
                # Calculate averages
                for char, values in avg_characteristics.items():
                    avg_characteristics[char] = round(np.mean(values), 3)
                
                analysis[pattern_type] = {
                    'count': len(patterns),
                    'average_characteristics': avg_characteristics
                }
        
        return analysis
    
    def _generate_voice_recommendations(self, stream: str, evolution: Dict) -> List[str]:
        """Generate specific recommendations for voice optimization"""
        
        recommendations = []
        
        # Based on current performance
        current_impact = evolution.get('current_impact_level', 0)
        
        if current_impact < 3:
            recommendations.append("🎯 Increase paradigm-shift language and unique phrasing")
            recommendations.append("💫 Integrate more personal narrative with technical concepts")
            
        if evolution.get('current_accessibility', 0) < 1:
            recommendations.append("🌉 Add more metaphors and analogies for technical concepts")
            
        if evolution.get('current_emotional_resonance', 0) < 2:
            recommendations.append("❤️ Increase emotional stakes and personal vulnerability")
            
        # Meta-optimization insights
        top_predictors = sorted(
            self.voice_data['meta_optimization']['engagement_predictors'].items(),
            key=lambda x: abs(x[1]), reverse=True
        )[:3]
        
        if top_predictors:
            top_predictor = top_predictors[0]
            recommendations.append(f"⚡ Focus on enhancing '{top_predictor[0]}' - highest impact predictor")
        
        return recommendations
    
    def _predict_next_level_potential(self, stream: str) -> Dict:
        """Predict what the next level of impact could look like"""
        
        current_evolution = self._calculate_voice_evolution(stream)
        
        # Predict potential improvements
        next_level = {}
        
        for metric, current_value in current_evolution.items():
            if isinstance(current_value, (int, float)):
                # Assume 20-50% improvement potential
                improvement_factor = 1.3
                next_level[f"potential_{metric}"] = round(current_value * improvement_factor, 2)
        
        return {
            'next_level_metrics': next_level,
            'breakthrough_potential': self._assess_breakthrough_potential(stream)
        }
    
    def _assess_breakthrough_potential(self, stream: str) -> str:
        """Assess potential for breakthrough-level impact"""
        
        stream_data = self.voice_data['voice_streams'][stream]
        
        # Count recent high-impact predictions
        recent_entries = []
        for context_entries in stream_data['contexts'].values():
            recent_entries.extend(context_entries[-3:])
        
        if not recent_entries:
            return "insufficient_data"
        
        high_impact_count = sum(1 for entry in recent_entries 
                              if entry['impact_prediction']['overall_impact_prediction'] > 7)
        
        if high_impact_count >= 2:
            return "high_breakthrough_potential"
        elif high_impact_count == 1:
            return "moderate_breakthrough_potential"
        else:
            return "developing_potential"

# Streamlit Interface
def main():
    st.title("🚀 Adaptive Voice Learning & Impact Optimization")
    st.markdown("*Learn your voice organically and optimize for maximum paradigm-shifting impact*")
    
    learner = AdaptiveVoiceLearner()
    
    # Sidebar for stream selection
    st.sidebar.header("🎯 Voice Stream")
    selected_stream = st.sidebar.selectbox(
        "Select Content Stream",
        ["personal_brand", "noeix_company"],
        format_func=lambda x: "🧠 Personal Brand" if x == "personal_brand" else "🏢 Noeix Company"
    )
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Learn Voice", 
        "📊 Voice Evolution", 
        "🎯 Optimization", 
        "🚀 Impact Prediction"
    ])
    
    with tab1:
        st.header("📝 Organic Voice Learning")
        
        # Content input
        content_input = st.text_area(
            "Share your content for organic voice learning:",
            height=200,
            placeholder="Paste any piece of your writing here. The system will learn your voice patterns and optimize for impact..."
        )
        
        # Metadata inputs
        col1, col2 = st.columns(2)
        with col1:
            content_type = st.selectbox(
                "Content Type",
                ["article", "social_post", "email", "presentation", "other"]
            )
        
        with col2:
            performance_known = st.checkbox("I know how this performed")
        
        # Performance feedback (if known)
        feedback = {}
        if performance_known:
            st.subheader("📈 Performance Feedback")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                feedback['high_engagement'] = st.checkbox("High Engagement")
                feedback['viral'] = st.checkbox("Went Viral")
            
            with col2:
                feedback['paradigm_shift'] = st.checkbox("Shifted Thinking")
                feedback['authority_building'] = st.checkbox("Built Authority")
            
            with col3:
                feedback['engagement_score'] = st.slider("Engagement Level", 1, 10, 5)
        
        if st.button("🧠 Learn from This Content", type="primary"):
            if content_input.strip():
                with st.spinner("Learning your voice patterns..."):
                    
                    # Analyze content
                    metadata = {
                        'stream': selected_stream,
                        'content_type': content_type,
                        'feedback_provided': performance_known
                    }
                    
                    learning_entry = learner.analyze_content_context(content_input, metadata)
                    
                    # Update learning system
                    learner.update_learning(learning_entry, feedback if performance_known else None)
                    
                    st.success("✅ Voice learning complete!")
                    
                    # Display analysis
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🎨 Voice Signature")
                        voice_sig = learning_entry['voice_signature']
                        
                        st.metric("Paradigm Shift Potential", f"{voice_sig['paradigm_shift_potential']}")
                        st.metric("Emotional Resonance", f"{voice_sig['emotional_resonance']:.2f}")
                        st.metric("Technical Accessibility", f"{voice_sig['technical_accessibility']:.2f}")
                        st.metric("Intellectual Honesty", f"{voice_sig['intellectual_honesty']}")
                        
                    with col2:
                        st.subheader("🚀 Impact Prediction")
                        impact = learning_entry['impact_prediction']
                        
                        st.metric("Viral Potential", f"{impact['viral_potential']:.1f}/10")
                        st.metric("Authority Building", f"{impact['authority_building']:.1f}/10")
                        st.metric("Paradigm Shift", f"{impact['paradigm_shift_potential']:.1f}/10")
                        st.metric("Overall Impact", f"{impact['overall_impact_prediction']:.1f}/10")
                    
                    # Context and unique elements
                    st.subheader("🎯 Content Analysis")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Context Type:**", learning_entry['context_type'])
                        if voice_sig['unique_phrasing']:
                            st.write("**Unique Phrases:**", ', '.join(voice_sig['unique_phrasing']))
                    
                    with col2:
                        st.write("**Narrative Integration:**", f"{voice_sig['narrative_integration']:.2f}")
                        st.write("**Vulnerability/Authority Ratio:**", f"{voice_sig['vulnerability_authority_ratio']:.2f}")
            
            else:
                st.error("Please enter some content to learn from.")
    
    with tab2:
        st.header("📊 Voice Evolution Analysis")
        
        stream_data = st.session_state.adaptive_voice_system['voice_streams'][selected_stream]
        
        if stream_data['evolution_timeline']:
            st.subheader(f"Evolution Timeline for {selected_stream.replace('_', ' ').title()}")
            
            # Show evolution metrics
            latest_evolution = stream_data['evolution_timeline'][-1]['voice_evolution']
            
            if latest_evolution:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Current Impact Level", f"{latest_evolution['current_impact_level']:.1f}/10")
                
                with col2:
                    st.metric("Technical Accessibility", f"{latest_evolution['current_accessibility']:.2f}")
                
                with col3:
                    st.metric("Emotional Resonance", f"{latest_evolution['current_emotional_resonance']:.2f}")
                
                with col4:
                    trend = latest_evolution['trend_direction']
                    trend_emoji = {"improving": "📈", "stable": "➡️", "declining": "📉"}.get(trend, "❓")
                    st.metric("Trend", f"{trend_emoji} {trend.replace('_', ' ').title()}")
            
            # Context breakdown
            st.subheader("📝 Content Contexts")
            
            context_data = []
            for context, entries in stream_data['contexts'].items():
                if entries:
                    avg_impact = np.mean([e['impact_prediction']['overall_impact_prediction'] for e in entries])
                    context_data.append({
                        'Context': context.replace('_', ' ').title(),
                        'Samples': len(entries),
                        'Avg Impact': round(avg_impact, 1),
                        'Last Updated': entries[-1]['timestamp'][:16]
                    })
            
            if context_data:
                df = pd.DataFrame(context_data)
                st.dataframe(df, use_container_width=True)
        
        else:
            st.info(f"No voice evolution data for {selected_stream} yet. Start learning by adding content samples!")
    
    with tab3:
        st.header("🎯 Voice Optimization Insights")
        
        if st.button("🔍 Generate Optimization Analysis", type="primary"):
            with st.spinner("Analyzing optimization opportunities..."):
                insights = learner.generate_optimization_insights(selected_stream)
                
                # Current voice state
                st.subheader("📊 Current Voice State")
                current_state = insights['current_voice_state']
                
                if current_state:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        for metric, value in current_state.items():
                            if isinstance(value, (int, float)):
                                st.metric(metric.replace('_', ' ').title(), f"{value}")
                    
                    with col2:
                        st.write("**Trend Direction:**", current_state.get('trend_direction', 'Unknown').replace('_', ' ').title())
                
                # Successful patterns analysis
                st.subheader("🏆 Successful Patterns Analysis")
                patterns_analysis = insights['successful_patterns_analysis']
                
                if patterns_analysis:
                    for pattern_type, analysis in patterns_analysis.items():
                        st.write(f"**{pattern_type.replace('_', ' ').title()}** ({analysis['count']} samples)")
                        
                        if analysis['average_characteristics']:
                            characteristics_text = []
                            for char, value in analysis['average_characteristics'].items():
                                characteristics_text.append(f"{char.replace('_', ' ').title()}: {value}")
                            st.write("- " + " | ".join(characteristics_text))
                else:
                    st.info("No performance feedback data available yet. Add content with performance feedback to see successful patterns.")
                
                # Optimization recommendations
                st.subheader("🚀 Optimization Recommendations")
                recommendations = insights['optimization_recommendations']
                
                for rec in recommendations:
                    st.write(f"- {rec}")
                
                # Next level potential
                st.subheader("⚡ Next Level Potential")
                next_level = insights['predicted_next_level']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Breakthrough Potential:**")
                    breakthrough = next_level['breakthrough_potential']
                    breakthrough_emoji = {
                        "high_breakthrough_potential": "🚀",
                        "moderate_breakthrough_potential": "📈", 
                        "developing_potential": "🌱",
                        "insufficient_data": "❓"
                    }.get(breakthrough, "❓")
                    st.write(f"{breakthrough_emoji} {breakthrough.replace('_', ' ').title()}")
                
                with col2:
                    st.write("**Predicted Next Level Metrics:**")
                    next_metrics = next_level['next_level_metrics']
                    for metric, value in next_metrics.items():
                        if isinstance(value, (int, float)):
                            st.write(f"- {metric.replace('potential_', '').replace('_', ' ').title()}: {value}")
    
    with tab4:
        st.header("🚀 Impact Prediction & Meta-Optimization")
        
        # Meta-optimization insights
        st.subheader("🧠 Meta-Learning Insights")
        
        meta_data = st.session_state.adaptive_voice_system['meta_optimization']
        
        if meta_data['engagement_predictors']:
            st.write("**Top Impact Predictors:**")
            
            # Sort predictors by absolute correlation strength
            sorted_predictors = sorted(
                meta_data['engagement_predictors'].items(),
                key=lambda x: abs(x[1]), 
                reverse=True
            )[:10]
            
            predictor_data = []
            for predictor, correlation in sorted_predictors:
                predictor_data.append({
                    'Voice Characteristic': predictor.replace('_', ' ').title(),
                    'Impact Correlation': round(correlation, 3),
                    'Strength': 'High' if abs(correlation) > 2 else 'Medium' if abs(correlation) > 1 else 'Low'
                })
            
            if predictor_data:
                df = pd.DataFrame(predictor_data)
                st.dataframe(df, use_container_width=True)
        
        else:
            st.info("Meta-learning data will appear as you add more content samples with performance feedback.")
        
        # Voice comparison tool
        st.subheader("🔄 Cross-Stream Voice Comparison")
        
        if st.button("📊 Compare Voice Streams"):
            personal_data = st.session_state.adaptive_voice_system['voice_streams']['personal_brand']
            noeix_data = st.session_state.adaptive_voice_system['voice_streams']['noeix_company']
            
            comparison_data = []
            
            for stream_name, stream_data in [('Personal Brand', personal_data), ('Noeix Company', noeix_data)]:
                if stream_data['evolution_timeline']:
                    latest_evolution = stream_data['evolution_timeline'][-1]['voice_evolution']
                    
                    if latest_evolution:
                        comparison_data.append({
                            'Stream': stream_name,
                            'Impact Level': latest_evolution.get('current_impact_level', 0),
                            'Accessibility': latest_evolution.get('current_accessibility', 0),
                            'Emotional Resonance': latest_evolution.get('current_emotional_resonance', 0),
                            'Trend': latest_evolution.get('trend_direction', 'unknown').replace('_', ' ').title(),
                            'Total Samples': sum(len(entries) for entries in stream_data['contexts'].values())
                        })
            
            if comparison_data:
                df = pd.DataFrame(comparison_data)
                st.dataframe(df, use_container_width=True)
                
                # Voice optimization suggestions
                st.subheader("🎯 Cross-Stream Optimization Opportunities")
                
                if len(comparison_data) == 2:
                    personal_impact = comparison_data[0]['Impact Level']
                    noeix_impact = comparison_data[1]['Impact Level']
                    
                    if personal_impact > noeix_impact:
                        st.write("💡 **Suggestion:** Transfer successful personal narrative techniques to Noeix content")
                    elif noeix_impact > personal_impact:
                        st.write("💡 **Suggestion:** Apply Noeix's authoritative voice patterns to personal content")
                    else:
                        st.write("💡 **Suggestion:** Both streams performing similarly - experiment with hybrid approaches")
            
            else:
                st.info("Need content samples from both streams to compare.")
        
        # Advanced impact modeling
        st.subheader("🎯 Advanced Impact Modeling")
        
        with st.expander("🔬 Experimental Features"):
            st.markdown("""
            **Future Impact Optimization Features:**
            
            🧬 **Genetic Voice Algorithms**: Combine successful patterns from different pieces
            
            🎭 **Contextual Voice Switching**: Automatically adapt voice for different audiences
            
            📈 **Predictive Viral Modeling**: Predict viral potential before publishing
            
            🌊 **Paradigm Shift Detection**: Identify content that could shift industry thinking
            
            🎪 **Controversy Optimization**: Balance thought-provoking vs. alienating content
            
            🚀 **Breakthrough Moment Prediction**: Identify when your voice is ready for major impact
            """)
    
    # Footer with learning statistics
    st.markdown("---")
    
    # Overall system statistics
    total_samples = 0
    total_streams = 0
    
    for stream_name, stream_data in st.session_state.adaptive_voice_system['voice_streams'].items():
        stream_samples = sum(len(entries) for entries in stream_data['contexts'].values())
        total_samples += stream_samples
        if stream_samples > 0:
            total_streams += 1
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📊 Total Learning Samples", total_samples)
    
    with col2:
        st.metric("🎯 Active Voice Streams", total_streams)
    
    with col3:
        predictor_count = len(st.session_state.adaptive_voice_system['meta_optimization']['engagement_predictors'])
        st.metric("🧠 Learned Predictors", predictor_count)
    
    st.markdown("*🚀 Adaptive Voice Learning System - Evolving your voice for maximum paradigm-shifting impact*")

if __name__ == "__main__":
    main()
  
