// Source-based slice around line 95
// Method: <com.google.common.collect.TreeTraverser: TreeTraverser using(Function)>

   * This is useful if the function instance already exists, or so that you can supply a lambda
   * expressions. If those circumstances don't apply, you probably don't need to use this; subclass
   * {@code TreeTraverser} and implement its {@link #children} method directly.
   *
   * @since 20.0
   * @deprecated Use {@link com.google.common.graph.Traverser#forTree} instead. If you are using a
   *     lambda, these methods have exactly the same signature.
   */
  @Deprecated
  public static <T> TreeTraverser<T> using(
      Function<T, ? extends Iterable<T>> nodeToChildrenFunction) {
    checkNotNull(nodeToChildrenFunction);
    return new TreeTraverser<T>() {
      @Override
      public Iterable<T> children(T root) {
        return nodeToChildrenFunction.apply(root);
      }
    };
  }

