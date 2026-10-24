# install the required libraries
# pip install -q -r ./requirements.txt
# openai
# langchain
# langchain-openai
# langchain-core

# streamlit
# tiktoken

import uuid
from datetime import datetime

import streamlit as st
from openai import OpenAI, AuthenticationError
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

DEFAULT_SYSTEM_PROMPT = 'You are a helpful assistant.'
MODELS = ['gpt-5.2', 'gpt-5-mini', 'gpt-4o', 'gpt-4o-mini']
SUGGESTIONS = [
    ('💡', 'Explain a concept', 'Explain how large language models work in simple terms.'),
    ('✍️', 'Write something', 'Write a professional email asking for a project deadline extension.'),
    ('🧑‍💻', 'Help me code', 'Write a Python function that checks whether a string is a palindrome.'),
    ('🗺️', 'Plan a trip', 'Plan a 3-day itinerary for a first-time visit to Istanbul.'),
]

st.set_page_config(
    page_title='Your Custom Assistant',
    page_icon='🤖',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ---------------------------------------------------------------------------
# Premium styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --bg: #0d0f14;
        --surface: #151821;
        --surface-2: #1c2030;
        --border: rgba(255, 255, 255, 0.08);
        --text: #e8eaf0;
        --muted: #8b90a0;
        --accent: #8b5cf6;
        --accent-2: #22d3ee;
    }

    html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea {
        font-family: 'Inter', sans-serif !important;
    }

    .stApp {
        background: radial-gradient(1200px 600px at 70% -10%, rgba(139, 92, 246, 0.12), transparent 60%),
                    radial-gradient(900px 500px at -10% 110%, rgba(34, 211, 238, 0.08), transparent 60%),
                    var(--bg);
        color: var(--text);
    }

    /* keep widgets dark regardless of the viewer's Streamlit theme */
    .stApp p, .stApp label, .stApp span, .stApp li, .stApp h1, .stApp h2, .stApp h3,
    .stApp [data-testid="stCaptionContainer"] { color: var(--text); }
    .stApp [data-testid="stCaptionContainer"] p { color: var(--muted); }
    .stApp a { color: var(--accent-2); }
    div[data-baseweb="input"], div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"], div[data-baseweb="select"] > div {
        background: var(--surface-2) !important;
        border-color: var(--border) !important;
    }
    .stApp input, .stApp textarea { color: var(--text) !important; caret-color: var(--accent); }
    div[data-testid="stExpander"] details {
        background: var(--surface-2); border: 1px solid var(--border); border-radius: 12px;
    }

    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }

    .block-container { max-width: 860px; padding-top: 2rem; padding-bottom: 6rem; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: var(--surface);
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        justify-content: flex-start;
        text-align: left;
        background: transparent;
        border: 1px solid transparent;
        color: var(--text);
        border-radius: 10px;
        padding: 0.45rem 0.75rem;
        font-size: 0.88rem;
        transition: background 0.15s ease, border-color 0.15s ease;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: var(--surface-2);
        border-color: var(--border);
    }
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: var(--surface-2);
        border-color: rgba(139, 92, 246, 0.45);
    }
    section[data-testid="stSidebar"] .st-key-new_chat button {
        justify-content: center;
        background: linear-gradient(135deg, var(--accent), #6d28d9);
        color: #fff;
        font-weight: 600;
        border: none;
    }

    .brand {
        display: flex; align-items: center; gap: 0.6rem;
        font-weight: 700; font-size: 1.1rem; margin-bottom: 0.75rem;
    }
    .brand-logo {
        width: 32px; height: 32px; border-radius: 9px;
        display: grid; place-items: center;
        background: linear-gradient(135deg, var(--accent), var(--accent-2));
        font-size: 1rem;
    }
    .section-label {
        color: var(--muted); font-size: 0.72rem; font-weight: 600;
        letter-spacing: 0.08em; text-transform: uppercase;
        margin: 1.1rem 0 0.4rem 0.2rem;
    }
    .empty-history { color: var(--muted); font-size: 0.85rem; padding: 0.3rem 0.2rem; }

    /* Chat messages */
    div[data-testid="stChatMessage"] {
        background: transparent;
        border-radius: 14px;
        padding: 0.9rem 1rem;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]),
    div[data-testid="stChatMessage"]:has(div[aria-label="Chat message from assistant"]) {
        background: var(--surface);
        border: 1px solid var(--border);
    }

    /* Chat input */
    div[data-testid="stChatInput"] {
        border-radius: 16px;
        border: 1px solid var(--border);
        background: var(--surface);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.35);
    }
    div[data-testid="stBottom"] > div { background: transparent; }

    /* Chat input + system role: jet black text on a light field */
    div[data-testid="stChatInput"],
    div[data-testid="stChatInput"] div[data-baseweb="textarea"],
    section[data-testid="stSidebar"] div[data-baseweb="textarea"] {
        background: #f4f5f8 !important;
    }
    div[data-testid="stChatInput"] textarea,
    section[data-testid="stSidebar"] .stTextArea textarea {
        background: #f4f5f8 !important;
        color: #0a0a0a !important;
        -webkit-text-fill-color: #0a0a0a;
        caret-color: #0a0a0a;
    }
    div[data-testid="stChatInput"] textarea::placeholder,
    section[data-testid="stSidebar"] .stTextArea textarea::placeholder {
        color: #6b7080 !important;
        -webkit-text-fill-color: #6b7080;
    }

    /* Hero / welcome */
    .hero { text-align: center; margin: 9vh 0 2rem 0; }
    .hero h1 {
        font-size: 2.6rem; font-weight: 700; margin-bottom: 0.4rem;
        background: linear-gradient(135deg, #fff 20%, var(--accent) 60%, var(--accent-2));
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero p { color: var(--muted); font-size: 1.02rem; }

    .main .stButton > button, div[data-testid="stMainBlockContainer"] .stButton > button {
        width: 100%;
        min-height: 72px;
        text-align: left;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        color: var(--text);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .main .stButton > button:hover, div[data-testid="stMainBlockContainer"] .stButton > button:hover {
        transform: translateY(-2px);
        border-color: rgba(139, 92, 246, 0.55);
        color: var(--text);
    }

    /* API key gate */
    .gate-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 2rem 2rem 0.5rem 2rem;
        margin-top: 10vh;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.45);
        text-align: center;
    }
    .gate-card h2 { margin: 0.8rem 0 0.3rem 0; font-weight: 700; }
    .gate-card p { color: var(--muted); font-size: 0.95rem; }
    .gate-logo {
        width: 56px; height: 56px; margin: 0 auto; border-radius: 16px;
        display: grid; place-items: center; font-size: 1.6rem;
        background: linear-gradient(135deg, var(--accent), var(--accent-2));
    }
    div[data-testid="stForm"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
    }
    div[data-testid="stFormSubmitButton"] > button {
        width: 100%;
        background: linear-gradient(135deg, var(--accent), #6d28d9);
        color: #fff; border: none; font-weight: 600; border-radius: 10px;
    }
    .gate-note { color: var(--muted); font-size: 0.8rem; text-align: center; margin-top: 0.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------
def new_conversation():
    conv_id = str(uuid.uuid4())
    st.session_state.conversations[conv_id] = {
        'title': 'New chat',
        'messages': [],
        'system': DEFAULT_SYSTEM_PROMPT,
        'created': datetime.now(),
    }
    st.session_state.current_id = conv_id


def validate_api_key(api_key):
    try:
        OpenAI(api_key=api_key).models.list()
        return True, None
    except AuthenticationError:
        return False, 'Invalid API key. Please check it and try again.'
    except Exception as e:
        return False, f'Could not verify the key: {e}'


if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'conversations' not in st.session_state:
    st.session_state.conversations = {}
if 'current_id' not in st.session_state:
    st.session_state.current_id = None


# ---------------------------------------------------------------------------
# API key gate: the user must enter their own OpenAI key before chatting
# ---------------------------------------------------------------------------
if not st.session_state.api_key:
    _, center, _ = st.columns([1, 1.4, 1])
    with center:
        st.markdown(
            """
            <div class="gate-card">
                <div class="gate-logo">🤖</div>
                <h2>Your Custom ChatGPT</h2>
                <p>Enter your OpenAI API key to start chatting.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.form('api_key_form', border=True):
            key_input = st.text_input('OpenAI API Key', type='password', placeholder='sk-...')
            submitted = st.form_submit_button('Continue →')

        if submitted:
            if not key_input.strip():
                st.error('Please enter your API key.')
            else:
                with st.spinner('Verifying your key ...'):
                    ok, error = validate_api_key(key_input.strip())
                if ok:
                    st.session_state.api_key = key_input.strip()
                    st.rerun()
                else:
                    st.error(error)

        st.markdown(
            '<div class="gate-note">🔒 Your key is kept only in this browser session and is never stored. '
            'Get one at <a href="https://platform.openai.com/api-keys" target="_blank">platform.openai.com</a>.</div>',
            unsafe_allow_html=True,
        )
    st.stop()


if not st.session_state.current_id:
    new_conversation()


# ---------------------------------------------------------------------------
# Sidebar: ChatGPT-style conversation history + settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-logo">🤖</div>Custom ChatGPT</div>', unsafe_allow_html=True)

    if st.button('＋  New chat', key='new_chat', type='secondary', width='stretch'):
        # reuse the current chat if it is still empty
        current = st.session_state.conversations[st.session_state.current_id]
        if current['messages']:
            new_conversation()
        st.rerun()

    st.markdown('<div class="section-label">Chats</div>', unsafe_allow_html=True)

    history = sorted(
        st.session_state.conversations.items(),
        key=lambda item: item[1]['created'],
        reverse=True,
    )
    history = [(cid, c) for cid, c in history if c['messages']]

    if not history:
        st.markdown('<div class="empty-history">No conversations yet.</div>', unsafe_allow_html=True)

    for conv_id, conv in history:
        is_active = conv_id == st.session_state.current_id
        col_title, col_delete = st.columns([5, 1])
        with col_title:
            if st.button(
                f'💬  {conv["title"]}',
                key=f'open_{conv_id}',
                type='primary' if is_active else 'secondary',
                width='stretch',
            ):
                st.session_state.current_id = conv_id
                st.rerun()
        with col_delete:
            if st.button('🗑', key=f'delete_{conv_id}', help='Delete chat'):
                del st.session_state.conversations[conv_id]
                if is_active:
                    st.session_state.current_id = None
                st.rerun()

    st.markdown('<div class="section-label">Settings</div>', unsafe_allow_html=True)
    with st.expander('⚙️  Model & behavior'):
        model_name = st.selectbox('Model', MODELS, index=0)
        temperature = st.slider('Temperature', 0.0, 1.0, 0.5, 0.1)
        current = st.session_state.conversations[st.session_state.current_id]
        current['system'] = st.text_area(
            'System role',
            value=current['system'],
            help='Instructions that shape how the assistant behaves in this chat.',
        ) or DEFAULT_SYSTEM_PROMPT

    st.markdown('<div class="section-label">Account</div>', unsafe_allow_html=True)
    st.caption(f'🔑  Key: sk-...{st.session_state.api_key[-4:]}')
    if st.button('↩  Change API key', key='logout'):
        st.session_state.api_key = None
        st.rerun()


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
conversation = st.session_state.conversations[st.session_state.current_id]
chat = ChatOpenAI(model_name=model_name, temperature=temperature, api_key=st.session_state.api_key)

prompt = st.chat_input('Message your assistant ...')
if 'pending_prompt' in st.session_state:
    prompt = st.session_state.pop('pending_prompt')

if not conversation['messages'] and not prompt:
    st.markdown(
        '<div class="hero"><h1>How can I help you today?</h1>'
        '<p>Ask anything, or pick a suggestion below to get started.</p></div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, (icon, label, text) in enumerate(SUGGESTIONS):
        with cols[i % 2]:
            if st.button(f'{icon}  **{label}**\n\n{text}', key=f'suggestion_{i}', width='stretch'):
                st.session_state.pending_prompt = text
                st.rerun()

for msg in conversation['messages']:
    if isinstance(msg, HumanMessage):
        with st.chat_message('user', avatar='🧑'):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message('assistant', avatar='🤖'):
            st.markdown(msg.content)

# if the user entered a question
if prompt:
    if not conversation['messages']:
        conversation['title'] = prompt[:32] + ('…' if len(prompt) > 32 else '')
        conversation['created'] = datetime.now()

    conversation['messages'].append(HumanMessage(content=prompt))
    with st.chat_message('user', avatar='🧑'):
        st.markdown(prompt)

    history_messages = [SystemMessage(content=conversation['system'])] + conversation['messages']

    with st.chat_message('assistant', avatar='🤖'):
        try:
            # stream the response token by token, like ChatGPT
            response = st.write_stream(chunk.content for chunk in chat.stream(history_messages))
        except AuthenticationError:
            conversation['messages'].pop()
            st.session_state.api_key = None
            st.error('Your API key was rejected. Please enter a valid key.')
            st.stop()
        except Exception as e:
            conversation['messages'].pop()
            st.error(f'Something went wrong: {e}')
            st.stop()

    conversation['messages'].append(AIMessage(content=response))
    # refresh so the new chat shows up in the sidebar history
    st.rerun()

# run the app: streamlit run ./custom_streamlit_app.py
