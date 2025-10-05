// Source-based slice around line 207
// Method: <com.google.common.collect.TreeTraverser: UnmodifiableIterator postOrderIterator(T)>

          public void accept(T t) {
            children(t).forEach(this);
            action.accept(t);
          }
        }.accept(root);
      }
    };
  }

  UnmodifiableIterator<T> postOrderIterator(T root) {
    return new PostOrderIterator(root);
  }

  private static final class PostOrderNode<T> {
    final T root;
    final Iterator<T> childIterator;

    PostOrderNode(T root, Iterator<T> childIterator) {
      this.root = checkNotNull(root);
      this.childIterator = checkNotNull(childIterator);
