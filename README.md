# Introdução ao TensorFlow
O Tensorflow é uma das bibliotecas mais amplamente utilizadas para implementar o aprendizado de máquina e outros algoritmos que envolvem grandes operações matemáticas. O Tensorflow foi desenvolvido pelo Google e é uma das bibliotecas de aprendizado de máquina mais populares no GitHub. O Google usa o Tensorflow para aprendizado de máquina em quase todos os aplicativos. 

Se você já usou o Google Photos ou o Google Voice Search, então já utlizou uma aplicação criada com a ajuda do TensorFlow. Vamos compreender os detalhes por trás do TensorFlow.
Matematicamente, um tensor é um vetor N-dimensional, significando que um tensor pode ser usado para representar conjuntos de dados N-dimensionais. 

Aqui está um exemplo:
![](/images/01_tensors_example.png?raw=true-small)

A figura acima mostra alguns tensores simplificados com dimensões mínimas. À medida que a dimensão continua crescendo, os dados se tornam mais e mais complexos. Por exemplo, se pegarmos um Tensor da forma (3x3), posso chamá-lo de matriz de 3 linhas e colunas. Se eu selecionar outro Tensor de forma (1000x3x3), posso chamá-lo como tensor ou conjunto de 1000 matrizes 3x3. Aqui chamamos (1000x3x3) como a forma ou dimensão do tensor resultante. Os tensores podem ser constantes ou variáveis.

![](/images/02_rank_tensors.png?raw=true-small)

# TensorFlow x NumPy
TensorFlow e NumPy são bastante semelhantes (ambos são bibliotecas de matriz N-d). NumPy é o pacote fundamental para computação científica com Python. Ele contém um poderoso objeto array N-dimensional, funções sofisticadas (broadcasting) e etc. Acredito que os usuários Python não podem viver sem o NumPy. 

O NumPy tem suporte a matriz N-d, mas não oferece métodos para criar funções de tensor e automaticamente computar derivadas, além de não ter suporte a GPU, e esta é uma das principais razões para a existência do TensorFlow. Abaixo uma comparação entre NumPy e TensorFlow, e você vai perceber que muitas palavras-chave são semelhantes.

![](/images/03_numpy_x_tensorflow.png?raw=true-small)

# Grafo Computacional

Agora sabemos o que o tensor realmente significa e é hora de entender o fluxo. Este fluxo refere-se a um grafo computacional ou simplesmente um grafo. Grafos computacionais são uma boa maneira de pensar em expressões matemáticas. O conceito de grafo foi introduzido por Leonhard Euler em 1736 para tentar resolver o problema das Pontes de Konigsberg. 

Grafos são modelos matemáticos para resolver problemas práticos do dia a dia, com várias aplicações no mundo real tais como: circuitos elétricos, redes de distribuição, relações de parentesco entre pessoas, análise de redes sociais, logística, redes de estradas, redes de computadores e muito mais. Grafos são muito usados para modelar problemas em computação.

Um Grafo é um modelo matemático que representa relações entre objetos. Um grafo G = (V, E) consiste de um conjunto de vértices V (também chamados de nós), ligados por um conjunto de bordas ou arestas E.

Considere o diagrama abaixo:
![](/images/04_graph.png?raw=true-small)

Existem três operações: duas adições e uma multiplicação. 

Ou seja:
* c = a+b
* d = b+1
* e = c∗d

Para criar um grafo computacional, fazemos cada uma dessas operações nos nós, juntamente com as variáveis de entrada. Quando o valor de um nó é a entrada para outro nó, uma seta vai de um para outro e temos nesse caso um grafo direcionado.

Esses tipos de grafos surgem o tempo todo em Ciência da Computação, especialmente ao falar sobre programas funcionais. Eles estão intimamente relacionados com as noções de grafos de dependência e grafos de chamadas. Eles também são a principal abstração por trás do popular framework de Deep Learning, o TensorFlow.

Para mais detalhes sobre grafos, leia um dos capítulos do Deep Learning Book:
http://deeplearningbook.com.br/algoritmo-backpropagation-parte1-grafos-computacionais-e-chain-rule/

Um grafo para execução de um modelo de Machine Learning pode ser bem grande e podemos executar sub-grafos (porções dos grafos) em dispositivos diferentes, como uma GPU. 

Exemplo:

![](/images/05_tensor_on_gpu.png?raw=true-small)

A figura acima explica a execução paralela de sub-grafos. Aqui estão 2 operações de multiplicação de matrizes, já que ambas estão no mesmo nível. Os nós são executados em gpu_0 e gpu_1 em paralelo.

# Modelo de Programação TensorFlow
O principal objetivo de um programa TensorFlow é expressar uma computação numérica como um grafo direcionado. A figura abaixo é um exemplo de grafo de computação, que representa o cálculo de h = ReLU (Wx + b). Este é um componente muito clássico em muitas redes neurais, que conduz a transformação linear dos dados de entrada e, em seguida, alimenta uma linearidade (função de ativação linear retificada, neste caso).

![](/images/06_model_tensorflow.png?raw=true-small)

O grafo acima representa um cálculo de fluxo de dados; cada nó está em operação com zero ou mais entradas e zero ou mais saídas. As arestas do grafo são tensores que fluem entre os nós. Os clientes geralmente constroem um grafo computacional usando uma das linguagens frontend suportadas como Python e C ++ e, em seguida, iniciam o grafo em uma sessão a ser executada (Session é uma noção muito importante no TensorFlow, que estudaremos na sequência).

Vamos ver o grafo computacional acima em detalhes. Truncamos o grafo e deixamos a parte acima do nó ReLU, que é exatamente o cálculo h = ReLU (Wx + b).

![](/images/07_relu_model.png?raw=true-small)

Podemos ver o grafo como um sistema, que tem entradas (os dados x), saída (h neste caso), variáveis com estado (W e b) e um monte de operações (matmul, add e ReLU). 

Deixe-me apresentar-lhe um por um:

* **Placeholders**: para alimentar a entrada para treinar o modelo ou fazer inferência, devemos ter uma porta de entrada para o grafo. Espaços reservados (Placeholders) são nós cujos valores são alimentados em tempo de execução. Normalmente, queremos alimentar entradas de dados, rótulos e hiper-parâmetros no modelo;

* **Variáveis**: quando treinamos um modelo, usamos variáveis para manter e atualizar parâmetros. Ao contrário de muitos tensores que fluem ao longo das margens do grafo, uma variável é um tipo especial de operação. Na maioria dos modelos de aprendizado de máquina, existem muitos parâmetros que temos que aprender, que são atualizados durante o treinamento. Variáveis são nós com estado que armazenam parâmetros e produzem seus valores atuais de tempos em tempos. Seus estados são mantidos em múltiplas execuções de um grafo. Por exemplo, os valores desses nós não serão atualizados até que uma etapa completa de treinamento usando um mini lote de dados seja concluída;

* **Operações matemáticas**: Neste grafo, existem três tipos de operações matemáticas. A operação MatMul multiplica dois valores de matriz; A operação Add adiciona elementos e a operação ReLU é ativada com a função linear retificada de elementos.

Variáveis devem ser explicitamente inicializadas. Quando criamos uma variável, passamos um tensor como seu valor inicial para o construtor variable (). O inicializador pode ser constantes, sequências e valores aleatórios. Neste caso, inicializamos o vetor de polarização b por constantes que são zeros e inicializamos a matriz de ponderações W por uniforme aleatório Observe que todos esses ops exigem que você especifique a forma dos tensores e que a forma se torne automaticamente a forma da variável , Neste caso, um tensor com forma (100,) e W é um tensor de classificação com forma (784, 100).

Para executar o cálculo, devemos lançar o grafo em um tf.Session. O que é uma sessão? Podemos entender uma sessão como um ambiente para executar o grafo. Na verdade, para fazer computação numérica eficiente em Python, normalmente usamos bibliotecas como o NumPy que realizam operações custosas computacionalmente, como a multiplicação de matrizes, usando código altamente eficiente implementado em outro idioma (C). 

Infelizmente, ainda há muita sobrecarga voltando para o Python em todas as operações. Essa sobrecarga é particularmente ruim se você quiser fazer cálculos em GPUs ou de maneira distribuída, onde pode haver um alto custo para transferir dados.
