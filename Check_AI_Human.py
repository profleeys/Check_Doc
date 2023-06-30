import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import matplotlib.pyplot as plt

tokenizer = AutoTokenizer.from_pretrained("Hello-SimpleAI/chatgpt-detector-roberta-chinese")
model = AutoModelForSequenceClassification.from_pretrained("Hello-SimpleAI/chatgpt-detector-roberta-chinese")

if __name__ == '__main__':
    st.title('AI文件鑑識系統')
    
    text = '''多項民調顯示，民眾黨總統參選人柯文哲領先國、民兩黨總統參選人；
台灣民意基金會董事長游盈隆認為，年輕選民不是今天才疏離民進黨；
把年輕選民逐漸疏離民進黨的現象，歸因於賴清德個人風格，甚至穿著，
基本上是犯了嚴重「見樹不見林」的謬誤。'''

    #context = st.text_input('請輸入資料', value = text)
    context = st.text_area('請輸入資料', value = text, height=160)
    
    if st.button('送出'):
        inputs = tokenizer(context, return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits
            
        #st.write(model.config.id2label)
        #st.write(logits.tolist()[0])

        predicted_class_id = logits.argmax().item()
        
        prob = round((logits.tolist()[0][predicted_class_id] + 5) * 10, 2)
        st.success('這份文件是' + model.config.id2label[predicted_class_id] + '寫的, 機率是' + str(prob) + '%!')
        
        prob = [0, 0]
        
        if predicted_class_id == 0:
            prob[0] = round((logits.tolist()[0][predicted_class_id] + 5) * 10, 2)
            prob[1] = 100 - round((logits.tolist()[0][predicted_class_id] + 5) * 10, 2)
        else:
            prob[0] = 100 - round((logits.tolist()[0][predicted_class_id] + 5) * 10, 2)
            prob[1] = round((logits.tolist()[0][predicted_class_id] + 5) * 10, 2)
        
        # 創建橫向長條圖
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.barh(['Human', 'GPT'], prob, color=['steelblue', 'orange'])
        
        # 設置圖表標籤
        ax.set_xlabel('Probability (%)', fontsize=8)
        #ax.set_ylabel('Model')

        # 設置圖表標題
        ax.set_title('Human vs GPT Probability', fontsize=8)

        # 設置橫軸範圍
        ax.set_xlim([0, 100])
        
        # 設置刻度標籤的字型大小
        ax.tick_params(axis='both', which='major', labelsize=8)

        # 顯示圖表
        st.pyplot(fig)