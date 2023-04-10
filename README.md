# DSL-23-1-modeling--Img2Music-Music-Recommendation-System-based-on-sentiment-analysis-
감정 기반 이미지-음악 추천시스템 

## 모델링 E조
- **팀명** : E = MusiC^2
- **팀원** : 8기 유채원 장준혁 최윤서 9기 김서진 서연우 

## Files
- image_to_music.py : final end-to-end file to get the result
- image_to_music_module.py : modules used in image_to_music.py
- image_to_sentiment
    - data
        - OASIS_with_minmaxscaling.csv
    - dataset.py : get data and transform it into runnable format
    - run.py : python file to receive arguments and run exp.py file
    - exp.py : python file actually run to train
    - utils.py : define functions for minor changes or transformations
    - vgg.sh : pretrained VGG19_bn model
- music_to_sentiment
    - arousal_valence_prediction.py : file to get music's sentiment
    - data
        - NRC-VAD-Lexicon.txt
        - da
        - song_lyric.csv
        - song_lyric_VA.csv
        - song_lyric_embedding_300.csv
        - song_lyric_embedding_pca_11.csv
        - song_music_8_characteristics.csv
        - song_normalized_VA_label.csv
    - preprocess : preprocessing modules for VA model
        - api_module.py
        - da
        - lyric_VA.py
        - lyrics_to_vector.py
        - music_preprocessing.py
        - music_to_csv.py
        - preprocess_module.py
    - result
        - arousal.pkl
---
## More Explanations
### 0. Task
- [Presentation pdf]()
- [Presentation Youtube]()
- 영상 안의 다양한 이미지에 어울리는 음악을 추천.
- Ekman's 6 feelings 에 기반한 Valence, Arousal 지표를 통해 이미지와 음악의 감성을 분석, image 2 music recommendation을 구현하고자 함. 
<img width="1209" alt="스크린샷 2023-04-08 오후 11 43 39" src="https://user-images.githubusercontent.com/116076204/230727394-5f8dc90c-c413-4d0d-8dca-ef8e03b7726e.png">

### 1. Data
- [OASIS](https://db.tt/yYTZYCga) (Open Affective Standardized Image Set) : 900개의 이미지에 대해 274명의 사람이 valence, arousal을 평가
- [Spotify API](https://developer.spotify.com/documentation/web-api) : Spotify data 크롤링을 통해 아티스트, 앨범, 곡명 및 valence, arousal을 포함한 음악의 정량적 지표를 수집
- [Musixmatch API](https://developer.musixmatch.com/) : Spotify API 기반으로 크롤링한 곡들의 가사를 수집
- [NRC VAD Lexicon](https://saifmohammad.com/WebPages/nrc-vad.html) : 캐나다 NRC 제작, 약 2만 여개의 단어에 대해 Valence, Arousal, Dominance 정보를 담고 있음.

### 2. Model
- Overview
    - <img width="1209" alt="스크린샷 2023-04-08 오후 11 24 05" src="https://user-images.githubusercontent.com/116076204/230726382-5d9df99f-01f3-48bc-9e87-f23065e95485.png">

- [Image2Emotion](./github/DSL-23-1-modeling-Img2Music/image_to_sentiment) : pretrained CNN model을 feature extractor로 사용해 valence, arousal 지표를 예측하도록 함
    <img width="755" alt="스크린샷 2023-04-08 오후 11 24 50" src="https://user-images.githubusercontent.com/116076204/230726422-21b1e00c-d442-4957-ae42-a6be68a8598e.png">
    - model
        - Feature extractor : VGG19_bn
        - Classifier : Linear layer
        - Loss : MSE loss (valence + arousal)
- [Music2Emotion](./github/DSL-23-1-modeling-Img2Music/music_to_sentiment)
    - Regression task to predict VA with Spotify data's columns
        - Used 8 features(danceability, key, loudness, mode, speechiness, instrumentalness, liveness, tempo)
        - Preprocessed with log normalization and min-max scaling
    - Lyric Embedding : Word2Vec + Weighted sum based on counts (NRC VAD Lexicon)
        - Used pretrained model(fse/word2vec-google-news-300(Hugging Face)) -> fine tuning -> PCA
    - AutoML : chose best 3 model and stacked them to make new model
        - Valence model : Gradient Boost Regressor + Random Forest Regressor + Extra Trees Regressor
            - Input : music + lyric embedding PCA + lyric VA
        - Arousal model: Gradient Boost Regressor + Random Forest Regressor + LGBM Regressor
            - Input : music + lyric VA
- [Image2Music](./github/DSL-23-1-modeling-Img2Music/image_to_music.py)
    - Used Euclidean distance as similarity measure(with sklearn)

### 3. Result

- End-to-end simulation code(image_to_music.py)
- result image
    <img width="1089" alt="스크린샷 2023-04-08 오후 11 26 10" src="https://user-images.githubusercontent.com/116076204/230726493-26d492c5-b6b9-4346-a624-0a9356fb6d4c.png">

### 4. References
1. Matt McVicar, Bruno Di Giorgi, Baris Dundar, and Matthias Mauch. (2021). Lyric document embeddings for music tagging
2. James A. Russell. (1980). A Circumplex Model of Affect 
3. Grekow, J. (2016). Music Emotion Maps in Arousal-Valence Space.
4. Fika Hastarita Rachman, RiyanartoSarno, ChastineFatichah. (2019). Song Emotion Detection Based on Arousal-Valencefrom Audio and Lyrics Using Rule Based Method
5. Minz Won, Justin Salamon, Nicholas J. Bryan, Gautham J. Mysore, Xavier Serra. (2021). Emotion Embedding Spaces for Matching Music to Stories
6. [Github hudsonbrendon/python-musixmatch](https://github.com/hudsonbrendon/python-musixmatch)
7. [스포티파이 API로 음악 분석하기](https://velog.io/@mare-solis/%EC%8A%A4%ED%8F%AC%ED%8B%B0%ED%8C%8C%EC%9D%B4-API%EB%A1%9C-%EC%9D%8C%EC%95%85-%EB%B6%84%EC%84%9D%ED%95%98%EA%B8%B0) 
8. [추천시스템 이해](https://dsbook.tistory.com/334)
9. Shuai Zhang, Lina Yao, Aixin Sun, and Yi Tay. 2018. Deep Learning based Recommender System: A Survey and New Perspectives. ACM Comput. Surv. 1, 1, Article 1 (July 2018), 35 pages.
10. Shankar, Devashish & Narumanchi, Sujay & Ananya, H & Kompalli, Pramod & Chaudhury, Krishnendu. (2017). Deep Learning based Large Scale Visual Recommendation and Search for E-Commerce.
11. [카카오 AI 추천 : 카카오의 콘텐츠 기반 필터링](https://tech.kakao.com/2021/12/27/content-based-filtering-in-kakao/)
12. 이승진. 2019. 음악 추천을 위한 가사정보 및 음악신호 기반 특성 탐색 연구. 서울대학교 융합과학기술대학원 석사학위 논문.
13. X. He and L. Deng, "Deep Learning for Image-to-Text Generation: A Technical Overview," in IEEE Signal Processing Magazine, vol. 34, no. 6, pp. 109-116, Nov. 2017, doi: 10.1109/MSP.2017.2741510.
14. Pan, Y., Mei, T., Yao, T., Li, H., & Rui, Y. (2015). Jointly Modeling Embedding and Translation to Bridge Video and Language. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 4594-4602.
15. Yin, Pei & Zhang, Liang. (2020). Image Recommendation Algorithm Based on Deep Learning. IEEE Access. PP. 1-1. 10.1109/ACCESS.2020.3007353. 
16. Lee, S. J., Seo, B.-G., & Park, D.-H. (2018). Development of Music Recommendation System based on Customer Sentiment Analysis. Journal of Intelligence and Information Systems, 24(4), 197–217. https://doi.org/10.13088/JIIS.2018.24.4.197
