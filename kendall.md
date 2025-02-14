| Networks                                   |                                    | N    | M      | Layers | Layer_Node_Nums | SAGE—myba—lr=0.005 |
| :----------------------------------------: | :--: | :----: | :----: | :----------------------------------------: | ------------------------------------------ | :----------------------------------------: |
| CS-Aarhus_Multiplex_Social                 | consists of five kinds of online and offline relationships (Facebook, Leisure, Work, Co-authorship, Lunch) between the employees of Computer Science department at Aarhus. | 61   | 620    | 5      | 60 / 32 / 25 / 47 / 60 | 0.8623 |
| Lazega-Law-Firm_Multiplex_Social           | consists of 3 kinds of (Co-work, Friendship and Advice) between partners and associates of a corporate law partnership. | 71   | 2223   | 3      | 71 / 71 / 70 | 0.8624 |
| HepatitusCVirus_Multiplex_Genetic          | We consider different types(Physical association, Direct interaction, Colocalization) of genetic interactions for organisms in the Biological General Repository for Interaction Datasets | 105  | 137    | 3      | 82 / 44 / 3 | 0.5418 |
| CKM-Physicians-Innovation_Multiplex_Social | They were concerned with the impact of network ties on the physicians' adoption of a new drug, tetracycline. | 246  | 1551   | 3      | 215 / 231 / 228 | 0.7778 |
| C.Elegans_Connectome_Multiplex.edges | **秀丽隐杆线虫连接组（Caenorhabditis elegans connectome）**，其多层网络由不同的突触连接类型构成：电突触（“ElectrJ”）、化学单体突触（“MonoSyn”）和化学多体突触（“PolySyn”） | 279 | 5863 | 3 | 253//260/278 |  |
| HumanHIV1_Multiplex_Genetic                | 1 Physical association 2Direct interaction 3Colocalization 4Association 5Suppressive genetic interaction defined by inequality | 1005 | 1355   | 4      | 758 / 380 / 34 / 21 | 0.4204 |
| Rattus_Multiplex_Genetic                   | 描述褐家鼠（Rattus Norvegicus）多层基因与蛋白质交互网络的数据集。 | 2640 | 4267   | 6      | 2035 / 1017 / 149/  39 / 8/ 15 | 0.4244 |
| C.Elegans_Multiplex_Genetic              | 描述秀丽隐杆线虫（Caenorhabditis Elegans）多层基因与蛋白质交互网络的数据集。 | 3879 | 8181   | 6      | 3126 / 239 / 1046 / 120 / 12 / 14 | 0.5446 |
| Arabidopsis_Multiplex_Genetic              | 拟南芥（Arabidopsis Thaliana）多层基因与蛋白质交互网络的数据集。 | 6980 | 186547 | 7      | 5493 / 2859 / 47 / 78 / 18 / 83 / 187 | 0.5413 |
| Drosophila_Multiplex_Genetic | 果蝇多层基因与蛋白质交互网络的数据集。 | 8215 | 43366 | 7 | 7356 / 839 / 755 / 2851 / 85 / 72 / 12 | 0.8136 |

|                  Networks                  |     DC     | Kshell |   GLSTM    |  RCNN   |  f-e   |  PRGC  |  SAGE_GAT  | Active Learning | SAGE_GAT_AL |
| :----------------------------------------: | :--------: | :----: | :--------: | :-----: | :----: | :----: | :--------: | --------------- | :---------: |
|         CS-Aarhus_Multiplex_Social         |  0.74754   | 0.7148 |   0.6339   | 0..6251 | 0.6514 | 0.5377 | **0.8350** | 0.8415(20)      |   0.8426    |
|      Lazega-Law-Firm_Multiplex_Social      |   0.7528   | 0.6901 |   0.4535   | 0.5662  | 0.8262 | 0.8101 | **0.8777** | 0.8857(30)      |   0.8229    |
|     HepatitusCVirus_Multiplex_Genetic      | **0.5344** | 0.4648 |   0.5081   | 0.4344  | 0.2571 | 0.056  |   0.5575   | 0.5828(6)       |   0.5410    |
| CKM-Physicians-Innovation_Multiplex_Social |   0.7066   | 0.6664 |   0.4882   | 0.6865  | 0.3749 | 0.3407 | **0.8142** | 0.8190(20)      |   0.8210    |
|    Celegans_Connectome_Multiplex.edges     |   0.6245   | 0.6366 |   0.4896   | 0.6201  | 0.6837 | 0.6576 | **0.8648** | 0.8670(3)       |   0.8312    |
|        HumanHIV1_Multiplex_Genetic         |   0.3666   | 0.3740 |   0.2722   | 0.2594  | 0.2920 | 0.1914 | **0.5105** | 0.5506(15)      |   0.3861    |
|          Rattus_Multiplex_Genetic          | **0.6140** | 0.4297 |   0.6067   | 0.0442  | 0.0623 | 0.1154 |   0.4367   | 0.6464(20)      |   0.5149    |
|         Celegans_Multiplex_Genetic         |   0.4403   | 0.4359 | **0.5120** | 0.0307  | 0.176  | 0.2301 |   0.4944   | 0.7206(10)      |   0.6460    |
|       Arabidopsis_Multiplex_Genetic        |   0.5090   | 0.5520 | **0.5703** | 0.2147  | 0.4821 | 0.3265 |   0.5860   | 0.8091(10)      |   0.7622    |
|        Drosophila_Multiplex_Genetic        |   0.7013   | 0.6942 |   0.6653   | 0.4210  | 0.4529 | 0.4639 |   0.8401   | 0.8341(5)       |   0.7407    |

# MGNN-AL: A Graph Neural Network with Active Learning for Multiplex Network Node Importance Prediction
