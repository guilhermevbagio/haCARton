# Validador de Segmentação

Treina um classificador para validar a qualidade da segmentação usando imagens lado a lado no formato:

`[imagem original] | [imagem segmentada]`

## Estrutura esperada

Dataset de entrada:

```text
dataset_rotulado/
  aprovar/
  rejeitar/
  revisar/
```

Split automático criado pelo treino, se necessário:

```text
dataset_validador/
  train/
    aprovar/
    rejeitar/
    revisar/
  val/
    aprovar/
    rejeitar/
    revisar/
```

Artefatos gerados:

```text
modelos/validador_segmentacao/
  validador_segmentacao.keras
  classes.json
```

## Treino

```bash
python modelos/validador_segmentacao/train_validator.py
```

O script:
- lê `dataset_rotulado/`
- cria `dataset_validador/` com split 80/20 se ainda não existir
- treina MobileNetV2 com transfer learning
- salva o melhor modelo e o mapeamento de classes

## Predição

```bash
python modelos/validador_segmentacao/predict_validator.py --image dataset_rotulado/aprovar/exemplo.png
```

A predição agora aplica também uma heurística leve de fora do domínio.
Se a imagem parecer uma foto comum com objeto central e fundo uniforme, a probabilidade de `rejeitar` recebe um reforço antes do resultado final.

## Importar negativos óbvios

```bash
python modelos/validador_segmentacao/import_negative_examples.py C:\caminho\paraotos_comuns --limit 50
```

Esse script pega fotos comuns, cria exemplos lado a lado com marcação visual de erro e salva tudo em:

```text
dataset_rotulado/rejeitar/
```

É útil para ensinar o validador a rejeitar entradas fora do domínio, como:
- comida
- selfie
- produto
- carro
- tela de celular

## Uso em código

```python
from modelos.validador_segmentacao.predict_validator import prever_qualidade_segmentacao

resultado = prever_qualidade_segmentacao("dataset_rotulado/aprovar/exemplo.png")
print(resultado)
```

Também existe a função:

```python
from modelos.validador_segmentacao.predict_validator import prever_qualidade_segmentacao_pil
```
