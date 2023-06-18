# Desenvolvimento de uma Aplicação de Aprendizado de Máquina para Classificação Multiclasse de MOOCs: Aprimorando a Avaliação da Dificuldade dos Cursos

Este é um projeto Python 3.10.9 que foca em utilizar algoritmos de aprendizado de máquina para classificação multi-classe de cursos online.

## Pré-requisitos

Para executar este projeto, será necessário ter as seguintes aplicações instaladas:

- Python 3.10.9
- Scikit-learn 1.2.1
- Jupyter Notebooks

## Instalação e Configuração

Siga as instruções abaixo para instalar e configurar o projeto:

1. Clone esse repositório dentro do Visual Studio Code ou utilize o comando:

`$ git clone https://github.com/joaopaulobarbosa/TCC_2023.git`

2. Baixe e instale o Anaconda, seguindo o tutorial adequado ao seu sistema operacional:

- [Windows](https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html#)
- [Linux](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)
- [macOS](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html)

3. Instale o Scikit Learn utilizando o tutorial de instalação para Windows, Linux e macOS disponível em [scikit-learn.org](https://scikit-learn.org/stable/install.html#installation-instructions).

4. Instale o Jupyter Notebook executando o seguinte comando no terminal:

`$ pip install notebook`

5. Instale as bibliotecas necessárias executando o seguinte comando no diretório raiz do projeto clonado:

`$ pip install -r requirements.txt`

6. Crie e ative um ambiente conda executando os seguintes comandos no terminal:

`$ conda create -n MOOCS_Classificator python=3.10.9`
`$ conda activate MOOCS_Classificator`

7. Crie um kernel Jupyter executando o seguinte comando no terminal:

`$ python -m ipykernel install --user --name MOOCS_Classificator --display-name <kernel_name>`

## Execução

Para executar o projeto, siga as instruções abaixo:

1. Abra o arquivo **_model_comparison.py_** no Visual Studio Code.
2. Na janela de exibição, no canto superior direito, clique no ícone de seta ao lado de _Run or debug_ e escolha a opção _Run this current file in interactive window_. O código será executado em uma janela interativa do Jupyter Notebook, permitindo a visualização de informações gráficas no output do código.
3. Repita o mesmo procedimento para executar os arquivos **_tunning_kneighborsclassifier.py_**, **_tunning_logisticregression.py_** e **_tunning_randomforestclassifier.py_**.

## Estrutura do Projeto

No diretório raiz, você encontrará os seguintes arquivos e diretórios:

- **_model_comparison.py_**: contém o código para comparação entre os modelos RandomForestClassifier, LogisticRegression e KNeighborsClassifier
