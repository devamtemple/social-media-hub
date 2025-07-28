import streamlit as st
import json
import re
import numpy as np
from datetime import datetime
from collections import defaultdict

# Initialize voice RAG in session state
if 'voice_rag' not in st.session_state:
    st.session_state.voice_rag = {
        'profiles': {
            'deva_temple': {'samples': [], 'patterns': {}},
            'noeix_company': {'samples': [], 'patterns': {}},
            'coherence_field': {'samples': [], 'patterns': {}}
        },
        'analysis_results': []
    }

def analyze_voice_sample(content, stream_name):
    """Analyze voice patterns in content sample"""
    
    # Basic metrics
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
    words = re.findall(r'\b\w+\b', content.lower())
    sentences = [s.strip() for s in re.split(r'[.!?]+', content) if s.strip()]
    
    # Technical terms
    tech_patterns = ['LLM', 'GPT', 'AI', 'alignment', 'consciousness', 'transformers', 'neural']
    technical_terms = [term for term in tech_patterns if term.lower() in content.lower()]
    
    # Signature phrases for Deva's voice
    deva_signatures = ['Truth™', 'The Answer™', 'walking all the way there, on my bare feet', 
                      'rabbit hole', 'funhouse mirror', 'collective descent into madness']
    found_signatures = [sig for sig in deva_signatures if sig in content]
    
    # Emotional vocabulary
    emotional_words = []
    emotions = ['grieved', 'cried', 'questioned', 'unsure', 'sanity', 'awe', 'wonder', 'longing']
    for emotion in emotions:
        if emotion in content.lower():
            emotional_words.append(emotion)
    
    # Authenticity markers
    em_dash_count = content.count('—')
    personal_narrative = any(marker in content for marker in 
                           ['I closed my computer', 'I grieved', 'my GPT', 'my own experience'])
    intellectual_honesty = any(marker in content for marker in 
                             ['I do not claim', "I don't know", 'unsure', 'uncertain'])
    
    # Calculate metrics
    vocab_complexity = len(set(words)) / len(words) if words else 0
    tech_density = len(technical_terms) / len(words) if words else 0
    avg_sentence_length = np.mean([len(s.split()) for s in sentences]) if sentences else 0
    
    analysis = {
        'timestamp': datetime.now().isoformat(),
        'stream': stream_name,
        'content_preview': content[:100] + '...' if len(content) > 100 else content,
        'metrics': {
            'paragraph_count': len(paragraphs),
            'word_count': len(words),
            'vocab_complexity': round(vocab_complexity, 3),
            'tech_density': round(tech_density, 4),
            'avg_sentence_length': round(avg_sentence_length, 1),
            'em_dash_count': em_dash_count,
            'personal_narrative': personal_narrative,
            'intellectual_honesty': intellectual_honesty
        },
        'patterns': {
            'technical_terms': technical_terms,
            'signature_phrases': found_signatures,
            'emotional_vocabulary': emotional_words
        },
        'authenticity_score': sum([
            em_dash_count == 0,  # No em-dashes (anti-LLM)
            personal_narrative,  # Personal integration
            intellectual_honesty,  # Intellectual humility
            len(found_signatures) > 0,  # Unique phrases
            len(emotional_words) > 0  # Emotional depth
        ])
    }
    
    return analysis

def save_analysis(analysis):
    """Save analysis to session state"""
    st.session_state.voice_rag['analysis_results'].append(analysis)
    
    # Update stream profile
    stream = analysis['stream']
    st.session_state.voice_rag['profiles'][stream]['samples'].append(analysis)
    
    # Update pattern aggregations
    patterns = st.session_state.voice_rag['profiles'][stream]['patterns']
    
    # Aggregate technical terms
    if 'technical_terms' not in patterns:
        patterns['technical_terms'] = defaultdict(int)
    for term in analysis['patterns']['technical_terms']:
        patterns['technical_terms'][term] += 1
    
    # Aggregate signature phrases
    if 'signature_phrases' not in patterns:
        patterns['signature_phrases'] = defaultdict(int)
    for phrase in analysis['patterns']['signature_phrases']:
        patterns['signature_phrases'][phrase] += 1

def generate_voice_prompt(stream_name):
    """Generate adaptation prompt based on learned voice patterns"""
    profile = st.session_state.voice_rag['profiles'][stream_name]
    
    if not profile['samples']:
        return "No voice samples analyzed yet. Please analyze some content first."
    
    # Aggregate metrics from all samples
    samples = profile['samples']
    avg_metrics = {
        'vocab_complexity': np.mean([s['metrics']['vocab_complexity'] for s in samples]),
        'tech_density': np.mean([s['metrics']['tech_density'] for s in samples]),
        'avg_sentence_length': np.mean([s['metrics']['avg_sentence_length'] for s in samples]),
        'authenticity_score': np.mean([s['authenticity_score'] for s in samples])
    }
    
    # Get most common patterns
    patterns = profile['patterns']
    top_tech_terms = sorted(patterns.get('technical_terms', {}).items(), 
                           key=lambda x: x[1], reverse=True)[:5]
    signature_phrases = list(patterns.get('signature_phrases', {}).keys())
    
    prompt = f"""
VOICE ADAPTATION CONTEXT for {stream_name}:

LEARNED VOICE METRICS:
- Vocabulary Complexity: {avg_metrics['vocab_complexity']:.3f}
- Technical Density: {avg_metrics['tech_density']:.4f}
- Average Sentence Length: {avg_metrics['avg_sentence_length']:.1f} words
- Authenticity Score: {avg_metrics['authenticity_score']:.1f}/5

SIGNATURE PATTERNS:
- Top Technical Terms: {', '.join([t[0] for t in top_tech_terms])}
- Unique Phrases: {', '.join(signature_phrases)}

VOICE REQUIREMENTS:
- NO em-dashes (use periods, commas, parentheses)
- Maintain personal narrative integration
- Include intellectual humility markers
- Preserve technical concepts woven into personal story
- Keep emotional vulnerability + intellectual authority balance

PLATFORM ADAPTATION GUIDELINES:
- Twitter: Reduce tech density to ~0.005, keep sentences under 15 words
- Medium: Maintain current complexity and technical integration
- LinkedIn: Professional tone but keep personal elements
"""
    
    return prompt

# Streamlit UI
st.title("🧠 Voice RAG System")
st.markdown("Learn and maintain authentic voice patterns across content streams")

# Sidebar for stream selection
st.sidebar.header("Voice Streams")
selected_stream = st.sidebar.selectbox(
    "Select Content Stream",
    ["deva_temple", "noeix_company", "coherence_field"]
)

# Main interface
tab1, tab2, tab3 = st.tabs(["📝 Analyze Content", "📊 Voice Patterns", "🎯 Generate Prompts"])

with tab1:
    st.header("Content Analysis")
    
    # Content input
    content_input = st.text_area(
        "Paste content sample for voice analysis:",
        height=200,
        placeholder="Paste a sample of writing to analyze voice patterns..."
    )
    
    if st.button("🔍 Analyze Voice Patterns", type="primary"):
        if content_input.strip():
            with st.spinner("Analyzing voice patterns..."):
                analysis = analyze_voice_sample(content_input, selected_stream)
                save_analysis(analysis)
                
                st.success("✅ Voice analysis complete!")
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Voice Metrics")
                    metrics = analysis['metrics']
                    st.metric("Vocabulary Complexity", f"{metrics['vocab_complexity']:.3f}")
                    st.metric("Technical Density", f"{metrics['tech_density']:.4f}")
                    st.metric("Avg Sentence Length", f"{metrics['avg_sentence_length']:.1f} words")
                    st.metric("Authenticity Score", f"{analysis['authenticity_score']}/5")
                
                with col2:
                    st.subheader("🎨 Voice Patterns")
                    if analysis['patterns']['technical_terms']:
                        st.write("**Technical Terms:**", ', '.join(analysis['patterns']['technical_terms']))
                    if analysis['patterns']['signature_phrases']:
                        st.write("**Signature Phrases:**", ', '.join(analysis['patterns']['signature_phrases']))
                    if analysis['patterns']['emotional_vocabulary']:
                        st.write("**Emotional Vocabulary:**", ', '.join(analysis['patterns']['emotional_vocabulary']))
                
                # Authenticity markers
                st.subheader("✅ Authenticity Markers")
                markers = [
                    ("No em-dashes", metrics['em_dash_count'] == 0),
                    ("Personal narrative", metrics['personal_narrative']),
                    ("Intellectual honesty", metrics['intellectual_honesty']),
                    ("Unique phrases", len(analysis['patterns']['signature_phrases']) > 0),
                    ("Emotional depth", len(analysis['patterns']['emotional_vocabulary']) > 0)
                ]
                
                for marker, present in markers:
                    st.write(f"{'✅' if present else '❌'} {marker}")
        
        else:
            st.error("Please enter some content to analyze.")

with tab2:
    st.header("📊 Voice Pattern Overview")
    
    profile = st.session_state.voice_rag['profiles'][selected_stream]
    
    if profile['samples']:
        st.subheader(f"Analysis History for {selected_stream}")
        
        # Show sample count and patterns
        st.metric("Total Samples Analyzed", len(profile['samples']))
        
        # Pattern frequency
        if profile['patterns']:
            col1, col2 = st.columns(2)
            
            with col1:
                if 'technical_terms' in profile['patterns']:
                    st.write("**Most Used Technical Terms:**")
                    tech_terms = dict(profile['patterns']['technical_terms'])
                    for term, count in sorted(tech_terms.items(), key=lambda x: x[1], reverse=True)[:10]:
                        st.write(f"- {term}: {count} times")
            
            with col2:
                if 'signature_phrases' in profile['patterns']:
                    st.write("**Signature Phrases Found:**")
                    for phrase in profile['patterns']['signature_phrases']:
                        st.write(f"- {phrase}")
        
        # Recent analyses
        st.subheader("Recent Analyses")
        for i, sample in enumerate(reversed(profile['samples'][-5:]), 1):
            with st.expander(f"Sample {len(profile['samples']) - i + 1} - {sample['timestamp'][:16]}"):
                st.write("**Content Preview:**", sample['content_preview'])
                st.write("**Authenticity Score:**", f"{sample['authenticity_score']}/5")
                st.write("**Technical Density:**", sample['metrics']['tech_density'])
                
    else:
        st.info(f"No voice samples analyzed for {selected_stream} yet. Use the 'Analyze Content' tab to get started.")

with tab3:
    st.header("🎯 Voice Adaptation Prompts")
    
    if st.button("📝 Generate Voice Prompt", type="primary"):
        prompt = generate_voice_prompt(selected_stream)
        
        st.subheader(f"Voice Adaptation Prompt for {selected_stream}")
        st.code(prompt, language="markdown")
        
        # Copy to clipboard button (display only)
        st.info("💡 Copy this prompt to use when adapting content for different platforms")

# Footer
st.markdown("---")
st.markdown("🧠 **Voice RAG System** - Learning authentic voice patterns through analysis")

# Debug info (optional)
if st.sidebar.checkbox("Show Debug Info"):
    st.sidebar.subheader("Debug Information")
    st.sidebar.json(st.session_state.voice_rag)
