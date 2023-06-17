# DESENVOLVIMENTO DE UMA APLICAÇÃO DE APRENDIZADO DE MÁQUINA PARA CLASSIFICAÇÃO MULTICLASSE DE MOOCS: APRIMORANDO A AVALIAÇÃO DA DIFICULDADE DOS CURSOS

Este é um projeto Python 3.10.9 que foca em utilizar algoritmos de aprendizado de máquina para classificação multiclasse de cursos online.

Para executar este projeto, será necessário ter as seguintes aplicações instaladas:
python 3.10.9
scikit_learn 1.2.1
jupyter notebooks

Abaixo, tutorial de instalação e configuração para o projeto:

1. Clone esse repositório dentro do Visual Studio Code ou utilize o comando:
$ git clone https://github.com/joaopaulobarbosa/TCC_2023.git

2. Baixar e instalar o Anaconda
  Segue tutorial para instalação em Windows:
    https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html#
  Segue tutorial para instalação on Linux: 
    https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html#
  Segue tutorial para instalação no macOS:
    https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html

3. Instale o Scikit Learn:
  Segue tutorial para instalação wm Windows, Linux e macOS
  https://scikit-learn.org/stable/install.html#installation-instructions

4. Instale o Jupyter Notebook:
  $ pip install notebook

5. Execute o arquivo requirements.txt para instalar as bibliotecas. No diretório raiz do projeto clonado:
  $ pip install -r requirements.txt

6. criar e ativar um ambiente conda:
  $ conda create -n MOOCS_Classificator python=3.10.9
  $ conda activate MOOCS_Classificator

7. Criar um kernel Jupyter
  $ python -m ipykernel install --user --name MOOCS_Classificator --display-name <kernel_name>

8. Abra o arquivo "model comparison roc auc.py" no VS Code Studio.
Na janela de exibição, no canto superior à direita, clique no ícone de seta ao lado de "Run or debug". Escolha a opção "Run this current file in interactive window". O código será executado em uma janela interativa do Jupyter notebook, o que vai possibilitar plotar informações gráficas no output do código.
O mesmo pode ser feito para executar os arquivos:
  tunning KNeighborsClassifier with f1 refit.py
  tunning LogisticRegression with f1 refit.py
  tunning RandomForestClassifier with f1 refit.py

No arquivo model comparison roc auc.py, se concentra o código para comparação entre os modelos RandomForestClassifier, LogisticRegression e KNeighborsClassifier.

Nos arquivos "tunning RandomForestClassifier with f1 refit.py", "tunning LogisticRegression with f1 refit.py" e "tunning KNeighborsClassifier with f1 refit.py" se concentram os códigos de hyperparameter tunning dos modelos.
