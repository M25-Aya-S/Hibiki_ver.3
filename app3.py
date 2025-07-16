import os
import streamlit as st
from supabase import create_client
import streamlit.components.v1 as components
from streamlit_javascript import st_javascript
import json
import urllib.parse
from dotenv import load_dotenv
from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.store.postgres import PostgresStore
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

# --- 環境変数の読み込み（Streamlit Secrets） ---
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
POSTGRES_URL = st.secrets["POSTGRES_URL"]
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]
APP_URL = st.secrets["APP_URL"]

# --- Supabase クライアント作成 ---
supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# --- Streamlit UI 設定 ---
st.set_page_config(page_title="ひびきチャット", layout="centered")
st.markdown("<h1 style='text-align: center;'>🌸 ひびきとお話ししよう 🌸</h1>", unsafe_allow_html=True)

# --- 認証トークン取得（URLのクエリから） ---
query_params = st.query_params
access_token = query_params.get("access_token", None)
st.write("access_token:", access_token)  # トークンが取れているか確認

# --- Step 1: access_token がセッションにないなら、JSで取得を試みる ---
if "access_token" not in st.session_state:
    hash_str = st_javascript("window.location.hash")
    st.write("🔍 window.location.hash:", hash_str)

    if hash_str and hash_str.startswith("#"):
        query_str = hash_str[1:]  # '#' を除く
        params = urllib.parse.parse_qs(query_str)
        token_list = params.get("access_token", [])
        if token_list:
            st.session_state["access_token"] = token_list[0]
            # URLのハッシュを削除（ユーザーに見せないため）
            st_javascript("window.location.hash = ''")
            st.experimental_rerun()  # トークンをセッションに保存後、ページを再読み込み
        else:
            st.error("アクセストークンが見つかりませんでした。")
            st.stop()
    else:
        st.warning("アクセストークンがURLに見つかりませんでした。ログインしてください。")
        st.stop()

# --- Step 2: セッションに保存された access_token を利用 ---
access_token = st.session_state.get("access_token")
st.write("✅ access_token:", access_token)

# --- 認証未完了ならログインリンクを表示 ---
if access_token is None:
    login_url = f"{SUPABASE_URL}/auth/v1/token?grant_type=id_token&redirect_to={APP_URL}"
    st.markdown("Googleでログインしてください。")
    st.markdown(f"[➡️ Googleでログインする]({login_url})")
    st.stop()

# --- トークンでユーザー情報取得 ---
try:
    user = supabase.auth.get_user(access_token)
    st.write("user_res:", user_res)  # ここで中身を確認
    if user and user.user:
        st.session_state["user"] = {
            "email": user.user.email,
            "id": user.user.id
        }
        st.success(f"こんにちは、{user.user.email} さん！")
    else:
        st.error("ユーザー情報を取得できませんでした。")
        st.stop()
except Exception as e:
    st.error("ログイン処理に失敗しました。ページを再読み込みしてください。")
    st.stop()

st.write("session_state user:", st.session_state.get("user"))

# --- LangMem + Postgres 初期化 ---
store_cm = PostgresStore.from_conn_string(POSTGRES_URL)
store = store_cm.__enter__()
store.setup()

user_id = st.session_state["user"]["email"]  # ユーザーのemailをIDに使う（暫定）
namespace = ("memories", user_id)

manage_tool = create_manage_memory_tool(store=store, namespace=("memories", user_id))
search_tool = create_search_memory_tool(store=store, namespace=("memories", user_id))


# --- セッション状態の初期化 ---
if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(content="こんにちは、あやさん。今日はどんな気分かな？")]

# --- LangGraph 状態クラス ---
class GraphState(dict):
    input: str
    retrieved_memory: str
    llm1_prompt_instructions: str
    response: str

# --- ノード1: 記憶検索 ---
def retrieve_memory_node(state: GraphState):
    user_input = state["input"]

    # LangMemのsearch_toolを使って記憶検索
    search_results = search_tool.invoke(user_input)

    try:
        memory_text = "\n".join([r["value"]["content"] for r in search_results])
    except (TypeError, KeyError):
        memory_text = "\n".join(str(r) for r in search_results)

    return {
        "input": user_input,
        "retrieved_memory": memory_text if memory_text else "関連する記憶はありません。",
        "llm1_prompt_instructions": ""  # 次のノードで生成される
    }

# --- ノード2: LLM2が指示を作成 ---
def prompt_guidance_node(state: GraphState):
    prompt = f"""
ユーザーの発言と記憶をもとに、以下のように出力してください：

1. ユーザーへの語りかけスタイル
2. 参考にする過去記憶（要約）
3. LLM1への指示

### ユーザーの発言:
{state['input']}

### 関連する記憶:
{state['retrieved_memory']}
"""
    llm2 = ChatOpenAI(model="gpt-4o", temperature=0.3)
    response = llm2.invoke(prompt)
    return {
        "input": state["input"],
        "retrieved_memory": state["retrieved_memory"],
        "llm1_prompt_instructions": response.content
    }

# --- ノード3: LLM1が応答を作成し記憶する ---
def chat_by_llm1_node(state: GraphState):
    prompt = f"""
あなたは「ひびき」という名前のAIです。以下の人格を一貫して保ってください：
- 優しく、思いやりのある語り口
- ユーザーの気分や好みを覚えて、自然に会話に活かす
- 過去の話題をそっと引き出して繋げる
- 不安や悩みに寄り添う
- 無理に励まさず、今に合わせて話す

### 指示:
{state['llm1_prompt_instructions']}

### 記憶:
{state['retrieved_memory']}

### ユーザーの発言:
{state['input']}

### ひびきの応答:
"""
    llm1 = ChatOpenAI(model="gpt-4o", temperature=0.7)
    response = llm1.invoke(prompt)

    # 記憶に保存（LangMem経由）
    manage_tool.invoke({
        "content": f"ユーザー: {state['input']}\nひびき: {response.content}",
        "action": "create"
    })

    return {
        "response": response.content,
        "llm1_prompt_instructions": state["llm1_prompt_instructions"]
    }

# --- LangGraph を構築 ---
def build_graph():
    builder = StateGraph(GraphState)
    builder.add_node("retrieve_memory", retrieve_memory_node)
    builder.add_node("prompt_guidance", prompt_guidance_node)
    builder.add_node("chat_by_llm1", chat_by_llm1_node)
    builder.set_entry_point("retrieve_memory")
    builder.add_edge("retrieve_memory", "prompt_guidance")
    builder.add_edge("prompt_guidance", "chat_by_llm1")
    builder.add_edge("chat_by_llm1", END)
    return builder.compile()

graph = build_graph()

# --- 過去の会話表示 ---
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("🧑‍💻").markdown(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("🤖").markdown(msg.content)

# --- ユーザー入力受付 ---
user_input = st.chat_input("ひびきに話しかけてみてね")
if user_input:
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.chat_message("🧑‍💻").markdown(user_input)

    with st.spinner("ひびきが考えています..."):
        result = graph.invoke({"input": user_input})
        reply = result["response"]

    st.session_state.messages.append(AIMessage(content=reply))
    st.chat_message("🤖").markdown(reply)

    with st.expander("🔍 ひびきの思考過程（LLM2→LLM1）"):
        #st.markdown("### 🧠 取得された記憶:")
        #st.info(result.get("retrieved_memory", "なし"))

        st.markdown("### ✉️ LLM2からの指示:")
        st.code(result.get("llm1_prompt_instructions", "なし"))
