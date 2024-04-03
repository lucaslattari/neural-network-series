# Importação de bibliotecas necessárias.
import os
import random
import tensorflow as tf
from tqdm import tqdm

# Definição do diretório onde as imagens estão armazenadas.
data_dir = "img_align_celeba"  # Caminho do diretório com as imagens.

# Lista todos os arquivos no diretório.
all_images = os.listdir(data_dir)

# Seleciona aleatoriamente 100.000 imagens do diretório.
selected_images = random.sample(all_images, 100000)


# Define uma função para criar um objeto tf.train.Feature que contém bytes.
def _bytes_feature(value):
    """Retorna um objeto Feature contendo uma lista de bytes a partir de um valor dado."""
    if isinstance(
        value, type(tf.constant(0))
    ):  # Converte de tf.Tensor para bytes, se necessário.
        value = value.numpy()  # Obtém o valor numpy do tensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# Define uma função para serializar uma imagem em uma string de bytes e retornar um exemplo
# TensorFlow.
def serialize_example(image_path):
    """Lê uma imagem, serializa para uma string de bytes e retorna um objeto Example serializado."""
    image_string = open(
        image_path, "rb"
    ).read()  # Lê a imagem como uma string de bytes.
    feature = {
        "image": _bytes_feature(image_string),  # Cria um dicionário de features.
    }
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature)
    )  # Cria um Example.
    return example_proto.SerializeToString()  # Serializa o Example para uma string.


# Define uma função para criar um arquivo TFRecord contendo as imagens selecionadas.
def create_tfrecord_file(selected_images, tfrecord_file_name, data_dir):
    """Cria um arquivo TFRecord a partir de uma lista de imagens."""
    with tf.io.TFRecordWriter(
        tfrecord_file_name
    ) as writer:  # Inicia um TFRecordWriter.
        for image_name in tqdm(
            selected_images, desc="Creating TFRecord"
        ):  # Loop através das imagens selecionadas com uma barra de progresso.
            image_path = os.path.join(
                data_dir, image_name
            )  # Constrói o caminho completo da imagem.
            example = serialize_example(
                image_path
            )  # Serializa a imagem para um Example.
            writer.write(example)  # Escreve o Example serializado no arquivo TFRecord.


# Nome do arquivo TFRecord a ser criado.
tfrecord_file_name = "celeba_subset.tfrecord"

# Chama a função para criar o arquivo TFRecord.
create_tfrecord_file(selected_images, tfrecord_file_name, data_dir)
