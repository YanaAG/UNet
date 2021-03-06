# Pytorch-UNet визуализация
Реализация сети [U-Net](https://arxiv.org/pdf/1505.04597.pdf) взята с  [репозитория](https://github.com/milesial/Pytorch-UNet).

Приложенная модель [CP5.pth](https://drive.google.com/open?id=1yk9yT0CiAYAhmjbA2EdYaZgEd2tbIVi9) была натренирована на наборе данных формата .jpeg, состоящего из N сканированных изображений мозга.

## Краткие теоретические сведения
![архитектура сети UNet](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)
Рисунок 1. Архитектура U-net (пример изображения с разрешением 32×32 пикселя — самым низким)
>
Каждый синий квадрат соответствует многоканальной карте свойств. Количество каналов приведено в верхней части квадрата. Размер x-y приведен в нижнем левом краю квадрата. Белые квадраты представляют собой копии карты свойств. Стрелки обозначают различные операции.
Архитектура сети приведена на рисунке 1. Она состоит из сужающегося пути (слева) и расширяющегося пути (справа). Сужающийся путь — типичная архитектуре сверточной нейронной сети. Он состоит из повторного применения двух сверток 3×3, за которыми следуют инит ReLU и операция максимального объединения (2×2 степени 2) для понижения разрешения.

На каждом этапе понижающей дискретизации каналы свойств удваиваются. Каждый шаг в расширяющемся пути состоит из операции повышающей дискретизации карты свойств, за которой следуют:

- свертка 2×2, которая уменьшает количество каналов свойств;
- объединение с соответствующим образом обрезанной картой свойств из стягивающегося пути;
- две 3×3 свертки, за которыми следует ReLU.

Первая часть делает проход вниз, это часть кода, где вы применяете блоки свертки с последующей понижающей дискретизацией maxpool для кодирования входного изображения в представления признаков на нескольких различных уровнях.
>
Вторая часть сети состоит из дискретизации и конкатенации, за которыми следуют регулярные операции свертки. Основная иделя повышение частоты дискретизации в CNN: мы расширяем размеры элементов до тех же размеров с помощью соответствующих блоков конкатенации слева. Вы можете увидеть серые и зеленые стрелки, где мы объединяем две карты объектов вместе. 
>
Основной вклад сети U-Net в этом смысле по сравнению с другими полностью сверточными сетями сегментации заключается в том, что при повышении частоты дискретизации и углублении в сети мы объединяем функции с более высоким разрешением из нижней части с функциями с повышенной дискретизацией, чтобы лучше локализовать и изучить представления с следующие свертки. Так как повышающая дискретизация является редкой операцией, нам необходим хороший предварительный анализ с более ранних этапов, чтобы лучше представить локализацию.
>
Изучив рисунок, вы можете заметить, что выходные размеры (388 x 388) не совпадают с исходными (572 x 572). Если вы хотите получить тот же размер, вы можете применить дополненные свертки, чтобы сохранить согласованность размеров на всех уровнях конкатенации.
>
Если имеются проблемы со сверточной арифметикой, можно прочитать дополнительную информацию [тут (Convolution arithmetic)](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)
>
## Требования
**Note : Use Python 3**
>
**Формат изображения : .jpeg**

### Визуализация
Чтобы запустить предсказание + визуализацию входного изображения из CLI, нужно ввести следующее:
>
`python predict.py --model CP5.pth —input OiDvTcZ.jpg —output output2.jpg —cpu —viz`

В качестве модели можно указать собственный экземпляр:
>
 `--model MODEL.pth`
 >
Чтобы использовать только cpu версию:
 >
 `--cpu`.

## Зависимости
This package depends on [pydensecrf](https://github.com/lucasb-eyer/pydensecrf), available via `pip install`.

### Описание
Каждый сверточный слой представляет собой контейнер из карт признаков, которые получаются посредством наложения набора фильтров на исходное изображение. 

В модуле unet_model.py происходит визуализация каждого слоя нейронной сети.
>
**Пример**
>
Исходное изображение:
>
![source](https://i.imgur.com/ZQCFgQF.png)
>
Слои (down1-down4):
>
<img src="https://i.imgur.com/L6FTW5Y.png" width="200"> <img src="https://i.imgur.com/m3GPMvF.png" width="200">
<img src="https://i.imgur.com/7t8mHKI.png" width="200"> <img src="https://i.imgur.com/YBSPJy3.png" width="200">

>
Поднимаемся (Upsampling):
>
<img src="https://i.imgur.com/dBSCHWn.png" width="200"> <img src="https://i.imgur.com/8QiDi6T.png" width="200">
<img src="https://i.imgur.com/c3LcOVX.png" width="200"> <img src="https://i.imgur.com/8cKhAyj.png" width="200">

>

inc/outc:
>
<img src="https://i.imgur.com/8cKhAyj.png" width="200"> <img src="https://i.imgur.com/7leoOW8.png" width="200">

>

### Добавление карты признаков к последнему слою сети
*В данной сети используется погружение на 4 слоя, поэтому, если входное изображение имеет размер 256х256, то карта признаков, которая будет добавлена, должна быть размером 16х16.*
*Каждый слой в 2 раза меньше предыдущего*

### Подготовка карты признаков
Рассмотрим на одном примере, как подготовить данные для добавления карты признаков.
>
По аннотации к снимкам находим срез, на котором видна опухоль, например:
>
![src](https://i.imgur.com/jeFgQAh.jpg)
>
На снимке необходимо выделить образование, которое нам нужно сегментировать:
>
![step1](https://i.imgur.com/994qh9D.jpg)
>
Далее разбиваем изображение с помощью сетки, размер которой соответствует слою, в который будем добавлять карту признаков:
>
![step2](https://i.imgur.com/JrIm6a1.png)
>
Составляем бинарную матрицу по данной картинке, соответствующую черныйм и белым областям.

#### Пример с добавленной картой признаков
Входное изображение имеет размер 256х256px.
На следующем слое получаем размер 128х128, длаее 64х64 и т.д.
Т.к. в данной сети используется 4 слоя, конечный размер карты признаков 16х16.
>
Исходное:
>
![source](https://i.imgur.com/BsEllyy.png)

>
Слои (down1-down4):
>
<img src="https://i.imgur.com/pnMmZQi.png" width="200"> <img src="https://i.imgur.com/ZyFUgAa.png" width="200">
<img src="https://i.imgur.com/yZNHSrI.png" width="200"> <img src="https://i.imgur.com/Jkim7Kg.png" width="200">

>
Upsampling:
>
<img src="https://i.imgur.com/2TXugY9.png" width="200"> <img src="https://i.imgur.com/tvYqiXw.png" width="200">
<img src="https://i.imgur.com/S8v78Nq.png" width="200"> <img src="https://i.imgur.com/4uKO2DI.png" width="200">

>
inc/outc:
>
<img src="https://i.imgur.com/JBo8wuO.png" width="200"> <img src="https://i.imgur.com/48rJXcB.png" width="200">
