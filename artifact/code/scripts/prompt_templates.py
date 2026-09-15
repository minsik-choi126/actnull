"""Per-dataset zero-shot prompt ensembles, as used by the task-arithmetic benchmark.

The shipped v7 evaluation used a three-template generic ensemble, which depresses
absolute accuracy well below the published numbers for this benchmark.  These are the
dataset-specific ensembles that benchmark uses.  Whether they are the right ones is
checked empirically: the zero-shot ViT-B/32 average they produce is compared against
the published value before any merge conclusion is drawn.
"""

GENERIC = (
    "a photo of a {}.", "a photo of the {}.", "a blurry photo of a {}.",
)

CARS = (
    'a photo of a {}.', 'a photo of the {}.', 'a photo of my {}.', 'i love my {}!',
    'a photo of my dirty {}.', 'a photo of my clean {}.', 'a photo of my new {}.',
    'a photo of my old {}.',
)

DTD = (
    'a photo of a {} texture.', 'a photo of a {} pattern.', 'a photo of a {} thing.',
    'a photo of a {} object.', 'a photo of the {} texture.', 'a photo of the {} pattern.',
    'a photo of the {} thing.', 'a photo of the {} object.',
)

EUROSAT = (
    'a centered satellite photo of {}.', 'a centered satellite photo of a {}.',
    'a centered satellite photo of the {}.',
)

GTSRB = (
    'a zoomed in photo of a "{}" traffic sign.',
    'a centered photo of a "{}" traffic sign.',
    'a close up photo of a "{}" traffic sign.',
)

DIGITS = ('a photo of the number: "{}".',)

RESISC45 = (
    'satellite imagery of {}.', 'aerial imagery of {}.', 'satellite photo of {}.',
    'aerial photo of {}.', 'satellite view of {}.', 'aerial view of {}.',
    'satellite imagery of a {}.', 'aerial imagery of a {}.', 'satellite photo of a {}.',
    'aerial photo of a {}.', 'satellite view of a {}.', 'aerial view of a {}.',
    'satellite imagery of the {}.', 'aerial imagery of the {}.',
    'satellite photo of the {}.', 'aerial photo of the {}.',
    'satellite view of the {}.', 'aerial view of the {}.',
)

SUN397 = ('a photo of a {}.', 'a photo of the {}.')

BY_TASK = {
    "dtd": DTD,
    "eurosat": EUROSAT,
    "gtsrb": GTSRB,
    "mnist": DIGITS,
    "resisc45": RESISC45,
    "stanford-cars": CARS,
    "sun397": SUN397,
    "svhn": DIGITS,
}
