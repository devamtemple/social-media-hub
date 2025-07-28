import streamlit as st
import json
import os
from datetime import datetime
import uuid

# Configure page
st.set_page_config(
    page_title="Content Creation Hub",
    page_icon="📝",
    layout="wide"
)

# Initialize session state for content storage
if 'content_drafts' not in st.session_state:
    st.session_state.content_drafts = {}

if 'current_draft_id' not in st.session_state:
    st.session_state.current_draft_id = None

# Content streams configuration
CONTENT_STREAMS = {
    "personal": {
        "name": "Personal Brand",
        "color": "#1f77b4",
        "description": "Your personal voice and expertise"
    },
    "noeix": {
        "name": "Noeix Company",
        "color": "#ff7f0e", 
        "description": "Company updates, product news, business insights"
    },
    "coherence_field": {
        "name": "Coherence Field",
        "color": "#2ca02c",
        "description": "Technical updates, AI insights, system capabilities"
    }
}

# Content categories
CONTENT_CATEGORIES = [
    "Technical Deep Dive",
    "Industry Analysis", 
    "Product Update",
    "Personal Insights",
    "Company News",
    "AI Research",
    "Opinion Piece",
    "Tutorial/How-to",
    "Case Study",
    "Philosophy",
    "Ethics",
    "Alignment",
    "Other"
]

def save_draft(draft_data):
    """Save draft to session state"""
    draft_id = draft_data.get('id', str(uuid.uuid4()))
    draft_data['id'] = draft_id
    draft_data['last_modified'] = datetime.now().isoformat()
    st.session_state.content_drafts[draft_id] = draft_data
    return draft_id

def load_draft(draft_id):
    """Load draft from session state"""
    return st.session_state.content_drafts.get(draft_id, {})

def create_new_draft():
    """Create a new empty draft"""
    new_draft = {
        'id': str(uuid.uuid4()),
        'title': '',
        'content': '',
        'stream': 'personal',
        'category': 'Other',
        'tags': [],
        'status': 'draft',
        'created': datetime.now().isoformat(),
        'last_modified': datetime.now().isoformat()
    }
    draft_id = save_draft(new_draft)
    st.session_state.current_draft_id = draft_id
    return draft_id

# Main app
st.title("📝 Content Creation Hub")
st.markdown("Create and manage content across your three streams")

# Sidebar for draft management
with st.sidebar:
    st.header("📂 Draft Management")
    
    # New draft button
    if st.button("✨ New Draft", type="primary", use_container_width=True):
        create_new_draft()
        st.rerun()
    
    # Draft list
    if st.session_state.content_drafts:
        st.subheader("Recent Drafts")
        for draft_id, draft in sorted(st.session_state.content_drafts.items(), 
                                    key=lambda x: x[1]['last_modified'], reverse=True):
            stream_color = CONTENT_STREAMS[draft['stream']]['color']
            title_preview = draft['title'][:30] + "..." if len(draft['title']) > 30 else draft['title']
            title_display = title_preview if title_preview else "Untitled Draft"
            
            if st.button(
                f"🎯 {title_display}",
                help=f"Stream: {CONTENT_STREAMS[draft['stream']]['name']}\nModified: {draft['last_modified'][:16]}",
                use_container_width=True,
                key=f"load_{draft_id}"
            ):
                st.session_state.current_draft_id = draft_id
                st.rerun()

# Main content area
if st.session_state.current_draft_id:
    current_draft = load_draft(st.session_state.current_draft_id)
    
    # Content creation form
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("📋 Content Settings")
        
        # Stream selector
        selected_stream = st.selectbox(
            "Content Stream",
            options=list(CONTENT_STREAMS.keys()),
            format_func=lambda x: CONTENT_STREAMS[x]['name'],
            index=list(CONTENT_STREAMS.keys()).index(current_draft.get('stream', 'personal')),
            help="Choose which voice this content represents"
        )
        
        # Show stream description
        stream_info = CONTENT_STREAMS[selected_stream]
        st.info(f"💡 {stream_info['description']}")
        
        # Category selector
        selected_category = st.selectbox(
            "Content Category",
            options=CONTENT_CATEGORIES,
            index=CONTENT_CATEGORIES.index(current_draft.get('category', 'Other'))
        )
        
        # Tags input
        tags_input = st.text_input(
            "Tags (comma-separated)",
            value=", ".join(current_draft.get('tags', [])),
            help="Add tags to help categorize and find your content"
        )
        
        # Parse tags
        tags = [tag.strip() for tag in tags_input.split(',') if tag.strip()]
        
        # Status indicator
        st.markdown("---")
        st.markdown(f"**Status:** {current_draft.get('status', 'draft').title()}")
        st.markdown(f"**Created:** {current_draft.get('created', 'Unknown')[:16]}")
        st.markdown(f"**Modified:** {current_draft.get('last_modified', 'Unknown')[:16]}")
    
    with col1:
        st.subheader("✍️ Content Editor")
        
        # Title input
        title = st.text_input(
            "Content Title",
            value=current_draft.get('title', ''),
            placeholder="Enter your content title...",
            help="This will be adapted for different platforms"
        )
        
        # Content editor
        content = st.text_area(
            "Content (Markdown supported)",
            value=current_draft.get('content', ''),
            height=400,
            placeholder="""Write your long-form content here...

You can use **markdown** formatting:
- **Bold text**
- *Italic text*
- ## Headers
- [Links](https://example.com)
- `code snippets`

This will be your master content that gets adapted for different platforms.""",
            help="Write in markdown - this will be your source content for all platforms"
        )
        
        # Live preview
        if content:
            with st.expander("👀 Preview", expanded=False):
                st.markdown(content)
        
        # Word count
        word_count = len(content.split()) if content else 0
        char_count = len(content) if content else 0
        st.caption(f"📊 {word_count} words • {char_count} characters")
        
        # Save button
        col_save, col_delete = st.columns([3, 1])
        
        with col_save:
            if st.button("💾 Save Draft", type="primary", use_container_width=True):
                updated_draft = {
                    'id': st.session_state.current_draft_id,
                    'title': title,
                    'content': content,
                    'stream': selected_stream,
                    'category': selected_category,
                    'tags': tags,
                    'status': 'draft',
                    'created': current_draft.get('created', datetime.now().isoformat()),
                }
                save_draft(updated_draft)
                st.success("✅ Draft saved!")
                st.rerun()
        
        with col_delete:
            if st.button("🗑️ Delete", type="secondary", use_container_width=True):
                if st.session_state.current_draft_id in st.session_state.content_drafts:
                    del st.session_state.content_drafts[st.session_state.current_draft_id]
                    st.session_state.current_draft_id = None
                    st.success("🗑️ Draft deleted!")
                    st.rerun()

else:
    # Welcome screen
    st.markdown("""
    ## Welcome to your Content Creation Hub! 🚀
    
    This is your central workspace for creating content across three streams:
    
    **🎯 Personal Brand** - Your personal voice and expertise  
    **🏢 Noeix Company** - Company updates, product news, business insights  
    **🤖 Coherence Field** - Technical updates, AI insights, system capabilities  
    
    ### Get Started:
    1. Click **"✨ New Draft"** in the sidebar to create your first piece of content
    2. Choose your content stream and category
    3. Write your long-form content using markdown
    4. Save your draft and move to the next stage: platform adaptation
    
    ### What's Next:
    Once you have content drafts, we'll build the platform adaptation engine to automatically format your content for:
    - **Long-form:** Medium, Substack, LessWrong, LinkedIn, Noeix website
    - **Short-form:** Facebook, X, Instagram, TikTok
    
    Ready to create some amazing content? 💪
    """)

# Footer
st.markdown("---")
st.markdown("🎯 **Content Creation Hub** - Part of your social media management system")
