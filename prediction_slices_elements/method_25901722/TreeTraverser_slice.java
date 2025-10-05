// Source-based slice around line 142
// Method: <com.google.common.collect.TreeTraverser: UnmodifiableIterator preOrderIterator(T)>

          public void accept(T t) {
            action.accept(t);
            children(t).forEach(this);
          }
        }.accept(root);
      }
    };
  }

  UnmodifiableIterator<T> preOrderIterator(T root) {
    return new PreOrderIterator(root);
  }

  private final class PreOrderIterator extends UnmodifiableIterator<T> {
    private final Deque<Iterator<T>> stack;

    PreOrderIterator(T root) {
      this.stack = new ArrayDeque<>();
      stack.addLast(singletonIterator(checkNotNull(root)));
    }
