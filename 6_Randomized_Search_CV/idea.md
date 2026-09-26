An online retailer wants to predict whether a website visitor will purchase a premium membership (1 = Purchase, 0 = No Purchase) based on session activity (time on site, pages viewed, scroll depth, items added to cart, device type).

Instead of evaluating every single combination like Grid Search, we will build a larger dictionary of options and use RandomizedSearchCV to sample a fixed budget of configurations randomly.

 train_model->save it -> Deploy & Predict