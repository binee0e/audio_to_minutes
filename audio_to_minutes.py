import openai
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
import tempfile
import io

st.header("회의록 생성기📝")
openai.api_key = st.secrets["OPENAI_API_KEY"]
uploaded_file = st.file_uploader("오디오 파일을 여기에 업로드하세요.", type=['wav', 'mp3', 'm4a'])

if uploaded_file is not None:
    with st.spinner('요약을 기다리는 중…'):
        # MP3 파일 로드
        audio_buffer = io.BytesIO(uploaded_file.read())
        audio_buffer.seek(0)

        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_file.write(audio_buffer.read())
            tmp_file_path = tmp_file.name

    # Whisper를 사용해 오디오를 텍스트로 변환
        with open(tmp_file_path, "rb") as audio_file:
            response = openai.Audio.transcribe(
                model="whisper-1",
                file=audio_file
            )
        transcript = response["text"]
        sum_list2=[]
        sum_list=[]
        sum_list.append(transcript[0:1800])
        sum_list.append(transcript[1700:3600])
        sum_list.append(transcript[3500:5400])
        sum_list.append(transcript[5300:7200])
        sum_list.append(transcript[7100:9000])
        sum_list.append(transcript[8900:10800])
        sum_list.append(transcript[10700:12000])
        sum_list.append(transcript[11900:13800])
        while sum_list[-1]=='':
            del sum_list[-1]
        if 'sum'not in st.session_state:
            st.session_state['sum'] =[]
                
        for i in sum_list:
                    
                my_tem_sum="""
                #지시문
                다음의 정보를 주요정보의 누락없이 요약해줘.
                
                #정보
                {info}
                #출력형식
                [회의의 주요정보들을 모두 포함한 요약]
                        """
                prompt_sum= PromptTemplate.from_template(my_tem_sum)
                sum_a=prompt_sum.format(info=i)
                llm_sum = ChatOpenAI(temperature=0.7,max_tokens=1000,model_name='gpt-3.5-turbo',
                                        openai_api_key = st.secrets["OPENAI_API_KEY"])
                sum_list2.append(llm_sum.predict(sum_a))
                summary='\n\n\n'.join(sum_list2)
        st.session_state['sum'].append(summary)
        my_tem_write_2="""#지시문
                너는 토론이나 회의 내용을 요약해 사용자들이 빠르게 읽을 수 있게 회의록을 작성해주는 역할을 할거야. 
                텍스트 파일을 받아서 내용을 읽은 후 해당 토론의 내용이 무엇인지를 요약한 후 토론의 내용을 출력형식에 맞춰서 출력해주면 돼.
                
                #정보
                {info2}
                #제약조건
                -맥락을 파악해서 A팀과 B팀을 명확하게 인지할것
                -회의에 참가한 인원을 명확하게 판단할것
                -토론 중 토론과 관계없는 잡담은 생략할것
                -회의록은 회의 타임라인을 한눈에 볼 수 있도록 작성할것.
                -Bullet Point를 이용해서 작성할것
                -구독, 좋아요, 알림설정 등의 회의와 관계없는 내용은 반드시 뺄 것
    """
        sumsum_write= PromptTemplate.from_template(my_tem_write_2)
        sum_b=sumsum_write.format(info2=st.session_state['sum'])

        llm_write_2 = ChatOpenAI(temperature=0.7,max_tokens=1000,model_name='gpt-3.5-turbo',
        openai_api_key = st.secrets["OPENAI_API_KEY"]
        st.success(f'{llm_write_2.predict(sum_b)}')
