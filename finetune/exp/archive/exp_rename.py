import pandas as pd 


cv = "/Users/anjandash/Desktop/JEMMA_CODE_COMPLETION_EXP/NEON/JEMMA_LOCALNESS_CODEKITTY/JEMMA_LOCALNESS_CK_MAIN/JEMMA_COMP_train_CODEKITTY_RF.csv"
cv2 = "/Users/anjandash/Desktop/JEMMA_CODE_COMPLETION_EXP/NEON/JEMMA_LOCALNESS_CODEKITTY/JEMMA_LOCALNESS_CK_MAIN/JEMMA_COMP_train_CODEKITTY_RFF.csv"
train_data = pd.read_csv(cv, header=0)


train_data.columns = train_data.columns.str.replace('method_tokens', 'text')
train_data.columns = train_data.columns.str.replace('call_label', 'labels')

train_data.to_csv(cv2, index=False)