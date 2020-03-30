# Introdução ao TensorFlow
O Tensorflow é uma das bibliotecas mais amplamente utilizadas para implementar o aprendizado de máquina e outros algoritmos que envolvem grandes operações matemáticas. O Tensorflow foi desenvolvido pelo Google e é uma das bibliotecas de aprendizado de máquina mais populares no GitHub. O Google usa o Tensorflow para aprendizado de máquina em quase todos os aplicativos. 

Se você já usou o Google Photos ou o Google Voice Search, então já utlizou uma aplicação criada com a ajuda do TensorFlow. Vamos compreender os detalhes por trás do TensorFlow.
Matematicamente, um tensor é um vetor N-dimensional, significando que um tensor pode ser usado para representar conjuntos de dados N-dimensionais. 

Aqui está um exemplo:
![](/images/01_tensors_example.png?raw=true-small)

# TensorFlow x NumPy
TensorFlow e NumPy são bastante semelhantes (ambos são bibliotecas de matriz N-d). NumPy é o pacote fundamental para computação científica com Python. Ele contém um poderoso objeto array N-dimensional, funções sofisticadas (broadcasting) e etc. Acredito que os usuários Python não podem viver sem o NumPy. 

O NumPy tem suporte a matriz N-d, mas não oferece métodos para criar funções de tensor e automaticamente computar derivadas, além de não ter suporte a GPU, e esta é uma das principais razões para a existência do TensorFlow. Abaixo uma comparação entre NumPy e TensorFlow, e você vai perceber que muitas palavras-chave são semelhantes.

![](/images/03_numpy_x_tensorflow.png?raw=true-small)
