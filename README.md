# 1. Project Title
COVID-19-LSTM-Malaysia

# 2. Project Description
This is an open data that provided by the MoH (Ministry of Health Malaysia) as to date on 27 July 2022. The task is to predict new cases from now onwards using Long Short Term Memory (LSTM) Deep Learning methodology

# 3. Data Description
Contains the ![data](https://github.com/GeoLai/COVID-19-LSTM-Malaysia/tree/main/dataset) to be used for model building and model testing. 

The dataset contains 2 .csv files for training & testing. The goal will be to predict new cases whether Malaysia should go into full lockdown, semi-lockdown or endemic phase.

Model is evaluated using Mean Absolute Percentage Error and Mean Squared Error.

# 4. How to Install and Run the Project
This project was run in Conda environment using Spyder IDE (Interactive Development Environment). Several essential libraries required to be installed prior to running the code. 

For computer that does not have GPU, you might want to use external workspace such as ![Google Colab](https://colab.research.google.com/?utm_source=scs-index) for running your scripts which no additional modules installation are required.

# 5. How to Use the Project
The full ![code](https://github.com/GeoLai/COVID-19-LSTM-Malaysia/blob/main/covid19-pred.py) can be viewed here as a reference. For cleaner code construction, I have written Long Short Term Memory Deep Learning architecture in a separate ![module](https://github.com/GeoLai/COVID-19-LSTM-Malaysia/blob/main/covid19_module.py) file which some of tuning can be done during model training.

Visuals are provided which were generated from data visualization of the data, training curves which displayed in Tensorboard, snippet of training scores, learning architecture, model parameters where located in the ![image](https://github.com/GeoLai/COVID-19-LSTM-Malaysia/tree/main/images) folder.

# 6. Include Credits
Credits to data owner, MoH (Ministry of Health Malaysia). ![Data](GitHub - MoH-Malaysia/covid19-public: Official data on the COVID-19 epidemic in Malaysia. Powered by CPRC, CPRC Hospital System, MKAK, and MySejahtera.) was gathered during COVID-19 pandemic

# 7. Add a License
No license

# 8. Badges
### These codes are powered by
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)
![Spyder](https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

### If you found this beneficial and help you in way or another. You may want to
![BuyMeAWanTanMee](https://img.shields.io/badge/Buy%20Me%20a%20Wan%20Tan%20Mee-ffdd00?style=for-the-badge&logo=buy-me-a-wantanmee&logoColor=black)

