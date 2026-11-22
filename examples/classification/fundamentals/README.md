# Foundations Examples

These examples establish the minimum logic of a classification experiment: define the target, create a split, fit a learner, generate class scores, and evaluate the result.

- [01-baseline-majority.md](01-baseline-majority.md) - `ClaMajority`: baseline classifier that always predicts the most frequent class observed during training.
- [02-interpretable-tree.md](02-interpretable-tree.md) - `ClaDtree`: decision tree classifier with an easy-to-interpret splitting structure.
- [21-tree-rpart.md](21-tree-rpart.md) - `ClaRpart`: CART classification tree with the `rpart` backend.
