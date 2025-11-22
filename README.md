# 🩺 Classificador de Condições Médicas por Sintomas (NLP/ML)

## 🎯 Objetivo

Construir um modelo de **Machine Learning (ML)** capaz de inferir uma condição médica (rótulo) a partir de uma lista de sintomas em texto (features). O projeto demonstra um fluxo de trabalho completo, desde a vetorização de texto até a serialização do modelo treinado para uso em produção.

## 🛠️ Tecnologias e Dependências

O projeto foi desenvolvido em Python e utiliza as seguintes bibliotecas:

* **Pandas:** Para manipulação e estruturação dos dados (DataFrame).
* **Scikit-learn:** A principal biblioteca para Machine Learning, utilizada para o vetorizador TF-IDF, o classificador SVM e as métricas de avaliação.
* **Joblib:** Utilizada para serializar (salvar) o modelo e o vetorizador em disco, permitindo que sejam carregados posteriormente sem a necessidade de retreinamento.

## 🧠 Metodologia e Processo (Workflow)

O fluxo de trabalho do classificador é dividido em quatro etapas principais:

### 1. Pré-processamento e Divisão
Os dados de sintomas são limpos (transformados em minúsculas) e o conjunto de dados é dividido em **treino** e **teste** para garantir a avaliação imparcial do modelo.

### 2. Vetorização (Feature Engineering)
A principal etapa de Processamento de Linguagem Natural (NLP) é a conversão do texto em um formato numérico. Utilizamos a técnica **TF-IDF**.

#### Fundamentos Algébricos e Estatísticos (TF-IDF)
A matriz TF-IDF (Term Frequency-Inverse Document Frequency) representa cada conjunto de sintomas como um vetor. O valor de cada dimensão nesse vetor é um peso que reflete a importância de um sintoma (termo $t$) para a condição (documento $d$) em relação a todo o conjunto de dados.

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

Onde:
* $\text{TF}(t, d)$: Frequência do termo $t$ no documento $d$ (sintoma na condição).
* $\text{IDF}(t)$: Penaliza termos muito comuns em todos os documentos (sintomas genéricos como "fever").
    $$\text{IDF}(t) = \log\left(\frac{N}{\text{DF}(t)}\right)$$
    ($N$ = número total de condições; $\text{DF}(t)$ = número de condições que contêm o termo $t$).

### 3. Treinamento do Classificador (SVM)
O modelo escolhido é o **Support Vector Machine (SVM)** com um **kernel linear**, conhecido por seu bom desempenho em espaços vetoriais de alta dimensionalidade (como a matriz TF-IDF).

#### Fundamentos Algébricos (SVM)
O SVM trabalha encontrando o **hiperplano** que melhor separa as classes no espaço vetorial. Para um modelo linear, o hiperplano é definido pela equação:
$$w \cdot x - b = 0$$
Onde:
* $x$ é um vetor de entrada (o vetor TF-IDF).
* $w$ é o vetor normal ao hiperplano.
* $b$ é o termo de interceptação (bias).
O algoritmo otimiza $w$ e $b$ para **maximizar a margem** entre o hiperplano e os pontos de dados mais próximos (vetores de suporte).

### 4. Serialização e Persistência
Após o treinamento, o modelo SVM (`svm_classifier.joblib`) e o vetorizador TF-IDF (`tfidf_vectorizer.joblib`) são salvos. Isso garante que a aplicação em produção possa carregar esses arquivos e fazer previsões instantâneas, sem precisar rodar o processo de treinamento novamente.

## ✅ Resultados (Conjunto de Dados de Exemplo)

A avaliação demonstra a prova de conceito do modelo, com separação perfeita das classes no conjunto de teste simulado:

| Métrica | Flu | Sinusitis |
| :---: | :---: | :---: |
| **Acurácia Geral** | **1.00** | |
| **Precisão** | 1.00 | 1.00 |
| **Recall** | 1.00 | 1.00 |
| **F1-Score** | 1.00 | 1.00 |

#### Previsões em Novos Dados

| Sintomas | Condição Prevista |
| :---: | :---: |
| `'fever body ache'` | **Flu** |
| `'nasal congestion headache'` | **Sinusitis** |

