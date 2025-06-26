# Beyond Closed-World Assumption: Open-Set Recognition of ECG Using a Fea-ture Fusion Network and Multi-Loss Open-set Framework

The electrocardiogram (ECG) signal is a non-invasive diagnostic tool based on recording the heart's electrical activity. Utilizing deep learning for its automatic classifica-tion and analysis is a crucial ap-proach for diagnosing cardiovascu-lar diseases. However, most current methods adhere to the "closed-world assumption" and ECG datasets typically contain only a limited number of known signal types, rendering models incapable of iden-tifying unknown categories. Fur-thermore, due to the specific charac-teristics of ECG signals, such as inter-individual heterogeneity, mod-el performance often degrades sig-nificantly when algorithms for rec-ognizing unknown classes are intro-duced. To address these limitations, we first decompose the ECG signal into time-varying and time-invariant features using the Fourier transform. We then propose two key compo-nents: 1) a Time-varying Time-invariant Multi-scale Bidirectional Feature Fusion Network, and 2) a Multi-loss Openmax Open-Set Recognition Framework. The framework integrates category cross-entropy loss, open-set loss, center loss, and Openmax, enabling the model to maintain classification accuracy while overcoming the constraints of the closed-world assumption. The network contains dedicated branches for extracting time-varying and time-invariant features, coupled with a bidirection-al feature fusion module. This de-sign allows the model to focus on capturing common features across different patients, mitigating the limitations imposed by ECG signal properties on open-set recognition. Experiments were conducted on three ECG datasets (MIT-BIH, CPSC 2018, CINC2017) and nine derived datasets constructed from them. On the most challenging de-rived dataset CPSC-U3, our method achieves a weighted-F1 score of 70.38%, exceeding the state-of-the-art method by 7.61%. Results demonstrate that the proposed mod-el and framework outperform exist-ing state-of-the-art methods in both tasks: recognizing unknown ECG classes and distinguishing known classes. 

![image](https://github.com/user-attachments/assets/66657d5b-cf5f-4216-a232-be3dafd71e77)



**Keywords**:
ECG classification
Open set recognition
Bidirectional fusion
Deep learning

If you find our model/method/dataset useful, please cite our work：XXXXX

**The paper has not yet been accepted and only a demo version is currently available.**

 [Xidong Wu]([https://www.sciencedirect.com/science/article/pii/S0893608024004751](https://github.com/xidong66) 
Dongchen Wu
6.26

