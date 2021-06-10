## Model classes

Any model class in the ~/mdls folder should have the following properties:

1. The main class should be called `model()`
2. The `__init__` must always have an `(encoder,di_model)` argument
3. `encoder` should have a `transform_X` and a `transform_y` attribute
4. The main class should have a `fit(X,y)` and `predict(X)` attribute
