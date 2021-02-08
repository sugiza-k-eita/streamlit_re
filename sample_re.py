import streamlit as st
import numpy as np 
import pandas as pd 
#from PIL import Image
#import time
#↑画像を読み取るためのライブラリ
import base64
from io import BytesIO
from sklearn import linear_model
import sklearn.model_selection
model = linear_model.LinearRegression()


def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1') # <--- here
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = to_excel(df)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="sample.xlsx">Download excel file</a>' # decode b'abc' => abc

df = pd.read_excel("sample.xlsx")
st.markdown(get_table_download_link(df), unsafe_allow_html=True)

st.title("輸出量から輸入量を推定する")
st.header("重回帰を使う")

uploaded_file = st.file_uploader("Choose an excel file", type="xlsx")
#サンプルファイルを用意しておく

if st.button('go!'):
        
    all_sheet = pd.ExcelFile(uploaded_file)   
    df = pd.read_excel(all_sheet)
    df2 = df.transpose()
    df2 =df2.reset_index()
    df2.columns = ["年月","輸入量","輸出量"]
    df2= df2.drop(index=0)
    df2["年月"]=pd.to_datetime(df2["年月"])
    df2["年月"]=df2["年月"].dt.strftime("%Y%m")

    tmp = df2["輸入量"]-df2["輸出量"]
    import_data = df2["輸入量"]
    export_data = df2["輸出量"]
    df2["在庫量"] = 0

    flg = df2["在庫量"].copy() 
    flg.loc[1]=200
    for i in range(2,len(flg)+1):
        flg.loc[i] = tmp.loc[i]+flg.loc[i-1]
    df2["在庫量"]=flg.copy() 
    #st.table(df)
    
    year_month=list(df2["年月"].unique())
    predict_data = pd.DataFrame()
    for i in range(6, len(df2)):
        tmp = df2.loc[df2["年月"]==year_month[i]].copy()
        tmp.rename(columns={"在庫量":"当月在庫量"}, inplace=True)
        tmp = tmp.reset_index(drop=True)
        for j in range(1,6):
            tmp_before = df2.loc[df2["年月"]==year_month[i-j]].copy()
            del tmp_before["年月"]
            del tmp_before["輸入量"]
            del tmp_before["在庫量"]
            tmp_before.rename(columns={"輸出量":"輸出量{}ヶ月前".format(j)}, inplace=True)
            tmp_before=tmp_before.reset_index(drop=True)
            tmp=tmp.join(tmp_before, how='left')
        predict_data=pd.concat([predict_data, tmp], ignore_index=True)
    df2_drop=df2.copy()
    del df2_drop["年月"]
    st.table(df2)
    st.line_chart(df2_drop)

    X = predict_data[["輸出量1ヶ月前","輸出量2ヶ月前","輸出量3ヶ月前","輸出量4ヶ月前","輸出量5ヶ月前"]]
    y = predict_data["輸出量"]
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y)
    model.fit(X_train, y_train)

    one_month_ago = predict_data["輸出量1ヶ月前"]
    two_month_ago = predict_data["輸出量2ヶ月前"]
    three_month_ago = predict_data["輸出量3ヶ月前"]
    four_month_ago = predict_data["輸出量4ヶ月前"]
    five_month_ago = predict_data["輸出量5ヶ月前"]
    now_stock =predict_data["当月在庫量"]

    "輸出量を予測します"

    x1=[one_month_ago.loc[len(predict_data)-1], two_month_ago.loc[len(predict_data)-1], three_month_ago.loc[len(predict_data)-1], four_month_ago.loc[len(predict_data)-1], five_month_ago.loc[len(predict_data)-1]]
    x2=[900,1500,1300,1000,600]
    x_pred=[x1,x2]
    #st.write(model.predict(x_pred))
    export_predict=model.predict(x_pred)
    st.write(export_predict[0])

    import_predict = model.predict(x_pred)[0] + 500 - now_stock.loc[len(predict_data)-1]
    "来月の輸入量予測"
    st.write(import_predict)

