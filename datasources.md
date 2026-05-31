# Fuentes de Datos para Entrenamiento de Transformers

## 1. Traducción Automática (NMT)
Ideal para el Transformer original (Encoder-Decoder).

*   **Multi30k**: El "Hello World" de la traducción. Pequeño y rápido.
    *   *Carga:* `load_dataset("bentrevett/multi30k")`
*   **WMT14 (English-German)**: El estándar de la industria.
    *   *Carga:* `load_dataset("wmt14", "de-en")`
*   **OPUS-100**: Corpus multilingüe masivo.
    *   *Carga:* `load_dataset("opus100", "en-es")`

## 2. Modelado de Lenguaje Causal (Tipo GPT)
Para entrenar un Decoder-only Transformer.

*   **WikiText-103**: Artículos de Wikipedia curados con dependencias a largo plazo.
    *   *Carga:* `load_dataset("wikitext", "wikitext-103-v1")`
*   **OpenWebText**: Clon abierto del dataset usado para GPT-2 (extraído de Reddit).
    *   *Carga:* `load_dataset("stas/openwebtext-10k")` (Versión pequeña para pruebas)
*   **The Pile (Subset)**: El dataset más diverso (libros, código, artículos científicos).
    *   *Carga:* `load_dataset("EleutherAI/the_pile_subset")`

## 3. Código Fuente (CodeGen)
Si buscas crear un asistente de programación.

*   **The Stack**: Dataset masivo de código con licencias permisivas.
    *   *Carga:* `load_dataset("bigcode/the-stack-dedup")`
*   **CodeSearchNet**: Pares de funciones y documentación.
    *   *Carga:* `load_dataset("code_search_net", "python")`

## 4. Dataset de Tokenización (Custom)
Si vas a entrenar tu propio Tokenizer desde cero, te recomiendo usar una muestra de Wikipedia en el idioma objetivo:
*   `load_dataset("wikipedia", "20220301.en")`
