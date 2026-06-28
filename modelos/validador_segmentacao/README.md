# Validador de Segmenta??o

Treina um classificador para validar a qualidade da segmenta??o usando imagens lado a lado no formato:

`[imagem original] | [imagem segmentada]`

## Estrutura esperada

Dataset de entrada:

```text
dataset_rotulado/
  aprovar/
  rejeitar/
  revisar/
```

Split autom?tico criado pelo treino, se necess?rio:

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
- l? `dataset_rotulado/`
- cria `dataset_validador/` com split 80/20 se ainda n?o existir
- treina MobileNetV2 com transfer learning
- salva o melhor modelo e o mapeamento de classes

## Predi??o

```bash
python modelos/validador_segmentacao/predict_validator.py --image dataset_rotulado/aprovar/exemplo.png
```

## Uso em c?digo

```python
from modelos.validador_segmentacao.predict_validator import prever_qualidade_segmentacao

resultado = prever_qualidade_segmentacao("dataset_rotulado/aprovar/exemplo.png")
print(resultado)
```

Tamb?m existe a fun??o:

```python
from modelos.validador_segmentacao.predict_validator import prever_qualidade_segmentacao_pil
```
