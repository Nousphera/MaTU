"""Dataset identities in the eight- and thirty-task vision benchmarks."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    title: str
    url: str
    torchvision: str | None = None


_TV = "https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.{}.html"
_SPECS = [
    ("mnist", "MNIST", _TV.format("MNIST"), "MNIST"),
    ("svhn", "SVHN", "http://ufldl.stanford.edu/housenumbers/", "SVHN"),
    ("gtsrb", "GTSRB", "https://benchmark.ini.rub.de/gtsrb_dataset.html", "GTSRB"),
    (
        "dtd",
        "Describable Textures",
        "https://www.robots.ox.ac.uk/~vgg/data/dtd/",
        "DTD",
    ),
    ("eurosat", "EuroSAT RGB", "https://github.com/phelber/EuroSAT", "EuroSAT"),
    (
        "cars",
        "Stanford Cars",
        "https://ai.stanford.edu/~jkrause/cars/car_dataset.html",
        None,
    ),
    ("sun397", "SUN397", "https://vision.princeton.edu/projects/2010/SUN/", "SUN397"),
    (
        "resisc45",
        "NWPU RESISC45",
        "https://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html",
        None,
    ),
    ("cifar10", "CIFAR-10", "https://www.cs.toronto.edu/~kriz/cifar.html", "CIFAR10"),
    (
        "cifar100",
        "CIFAR-100",
        "https://www.cs.toronto.edu/~kriz/cifar.html",
        "CIFAR100",
    ),
    (
        "fashionmnist",
        "Fashion-MNIST",
        "https://github.com/zalandoresearch/fashion-mnist",
        "FashionMNIST",
    ),
    (
        "emnist",
        "EMNIST Balanced",
        "https://www.nist.gov/itl/products-and-services/emnist-dataset",
        "EMNIST",
    ),
    (
        "food101",
        "Food-101",
        "https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/",
        "Food101",
    ),
    (
        "pets",
        "Oxford-IIIT Pet",
        "https://www.robots.ox.ac.uk/~vgg/data/pets/",
        "OxfordIIITPet",
    ),
    ("stl10", "STL-10", "https://cs.stanford.edu/~acoates/stl10/", "STL10"),
    ("animals10n", "Animal-10N", "http://dm.kaist.ac.kr/datasets/animal-10n/", None),
    ("beans", "Beans", "https://huggingface.co/datasets/AI-Lab-Makerere/beans", None),
    (
        "catsdogs",
        "Dogs vs Cats",
        "https://www.kaggle.com/competitions/dogs-vs-cats",
        None,
    ),
    (
        "cub200",
        "CUB-200-2011",
        "https://www.vision.caltech.edu/datasets/cub_200_2011/",
        None,
    ),
    (
        "dogs",
        "Stanford Dogs",
        "http://vision.stanford.edu/aditya86/ImageNetDogs/",
        None,
    ),
    (
        "flowers",
        "Flowers Recognition (5 classes)",
        "https://www.kaggle.com/datasets/alxmamaev/flowers-recognition",
        None,
    ),
    ("fruits", "Fruits-360", "https://www.kaggle.com/datasets/moltean/fruits", None),
    (
        "garbage",
        "Garbage Classification",
        "https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification",
        None,
    ),
    (
        "intel",
        "Intel Image Classification",
        "https://www.kaggle.com/datasets/puneet6060/intel-image-classification",
        None,
    ),
    ("kenyanfood", "KenyanFood13", "https://github.com/monajalal/Kenyan-Food", None),
    ("kvasir_v2", "Kvasir-V2", "https://datasets.simula.no/kvasir/", None),
    (
        "landscape",
        "Landscape Recognition",
        "https://www.kaggle.com/datasets/utkarshsaxenadn/landscape-recognition-image-dataset-12k-images",
        None,
    ),
    (
        "mangoleaf",
        "MangoLeafBD",
        "https://data.mendeley.com/datasets/hxsnvwty3r/1",
        None,
    ),
    (
        "vegetables",
        "Vegetable Image Dataset",
        "https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset",
        None,
    ),
    ("weather", "Weather Image Dataset", "https://doi.org/10.7910/DVN/M8JQCR", None),
]
DATASETS = {row[0]: DatasetSpec(*row) for row in _SPECS}
BENCHMARKS = {
    "small": ["mnist", "svhn", "gtsrb", "dtd", "eurosat"],
    "eight": ["sun397", "cars", "resisc45", "eurosat", "svhn", "gtsrb", "mnist", "dtd"],
    "thirty": [
        "animals10n",
        "cars",
        "cifar10",
        "cub200",
        "dtd",
        "eurosat",
        "fashionmnist",
        "fruits",
        "gtsrb",
        "kenyanfood",
        "landscape",
        "mnist",
        "resisc45",
        "sun397",
        "vegetables",
        "beans",
        "catsdogs",
        "cifar100",
        "dogs",
        "emnist",
        "flowers",
        "food101",
        "garbage",
        "intel",
        "kvasir_v2",
        "mangoleaf",
        "pets",
        "stl10",
        "svhn",
        "weather",
    ],
}
_ALIASES = {
    "stanfordcars": "cars",
    "stanford_cars": "cars",
    "cars196": "cars",
    "fmnist": "fashionmnist",
    "fashion_mnist": "fashionmnist",
    "kvasirv2": "kvasir_v2",
    "kvasir": "kvasir_v2",
    "cub_200_2011": "cub200",
    "oxford_pets": "pets",
    "oxfordiiitpet": "pets",
    "oxford_iiit_pet": "pets",
    "stanford_dogs": "dogs",
    "flowers5": "flowers",
    "animal10n": "animals10n",
    "kenyanfood13": "kenyanfood",
    "mangoleafbd": "mangoleaf",
    "fruits360": "fruits",
    "cifar_10": "cifar10",
    "cifar_100": "cifar100",
    "food_101": "food101",
}


def canonical_name(name):
    value = name.lower().replace("-", "_").replace(" ", "_")
    value = _ALIASES.get(value, value)
    if value not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {', '.join(DATASETS)}")
    return value
