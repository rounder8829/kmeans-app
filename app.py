import pandas as pd
import streamlit as st
from pandas.api.types import is_integer_dtype, is_float_dtype, is_object_dtype
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import platform
import os
import matplotlib.font_manager as fm

# 폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Linux':
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

@st.cache_data
def fontRegistered():
    font_dirs = [os.getcwd() + '/custom_fonts']
    font_files = fm.findSystemFonts(fontpaths=font_dirs)
    for font_file in font_files:
        fm.fontManager.addfont(font_file)
    fm._load_fontmanager(try_read_cache=False)

def preprocess_data(df, selected_columns):
    df_new = pd.DataFrame()
    for column in selected_columns:
        if is_integer_dtype(df[column]) or is_float_dtype(df[column]):
            df_new[column] = df[column]
        elif is_object_dtype(df[column]):
            if df[column].nunique() == 2:
                encoder = LabelEncoder()
                df_new[column] = encoder.fit_transform(df[column])
            else:
                ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
                column_names = sorted(df[column].unique())
                df_new[column_names] = ct.fit_transform(df[column].to_frame())
        else:
            st.warning(f'{column} 컬럼은 K-Means에 사용 불가하므로 제외합니다.')
    return df_new

def plot_elbow_method(wcss, max_k):
    fig, ax = plt.subplots()
    ax.plot(range(1, max_k+1), wcss)
    ax.set_title('The Elbow Method')
    ax.set_xlabel('클러스터 갯수')
    ax.set_ylabel('WCSS 값')
    return fig

def plot_clusters(df, df_new, kmeans):
    if df_new.shape[1] >= 2:
        fig, ax = plt.subplots()
        scatter = ax.scatter(df_new.iloc[:, 0], df_new.iloc[:, 1], c=kmeans.labels_, cmap='viridis')
        ax.set_title('Cluster Visualization')
        plt.colorbar(scatter)
        return fig
    return None

def main():
    plt.rc('font', family='NanumGothic')
    fontRegistered()

    st.title('K-Means Clustering App')

    file = st.file_uploader('CSV 파일 업로드', type=['csv'])

    if file is not None:
        try:
            df = pd.read_csv(file)
            st.dataframe(df.head())

            st.info('Nan 값이 있는 행을 삭제합니다.')
            st.dataframe(df.isna().sum())
            df.dropna(inplace=True)
            df.reset_index(drop=True, inplace=True)

            selected_columns = st.multiselect('K-Means 클러스터링에 사용할 컬럼을 선택해주세요.', df.columns)

            if len(selected_columns) == 0:
                st.warning('컬럼을 선택해주세요.')
                return

            df_new = preprocess_data(df, selected_columns)

            st.info('K-Means를 수행하기 위한 데이터프레임입니다.')
            st.dataframe(df_new)

            st.subheader('최적의 k값을 찾기 위해 WCSS를 구합니다.')

            max_k = min(10, df_new.shape[0]) if df_new.shape[0] >= 10 else df_new.shape[0]
            max_k = st.slider('K값 선택(최대 그룹갯수)', min_value=2, max_value=max_k)

            wcss = []
            for k in range(1, max_k+1):
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(df_new)
                wcss.append(kmeans.inertia_)

            fig1 = plot_elbow_method(wcss, max_k)
            st.pyplot(fig1)

            k = st.number_input('원하는 클러스터링(그룹) 갯수를 입력하세요.', min_value=2, max_value=max_k)

            kmeans = KMeans(n_clusters=k, random_state=42)
            df['Group'] = kmeans.fit_predict(df_new)

            st.info('그룹 정보가 저장되었습니다.')
            st.dataframe(df)

            fig2 = plot_clusters(df, df_new, kmeans)
            if fig2:
                st.pyplot(fig2)

        except Exception as e:
            st.error(f'오류가 발생했습니다: {str(e)}')

if __name__ == '__main__':
    main()
