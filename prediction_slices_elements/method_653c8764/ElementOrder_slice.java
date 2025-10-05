// Source-based slice around line 80
// Method: <com.google.common.graph.ElementOrder: ElementOrder unordered()>

  }

  private ElementOrder(Type type, @Nullable Comparator<T> comparator) {
    this.type = checkNotNull(type);
    this.comparator = comparator;
    checkState((type == Type.SORTED) == (comparator != null));
  }

  /** Returns an instance which specifies that no ordering is guaranteed. */
  public static <S> ElementOrder<S> unordered() {
    return new ElementOrder<>(Type.UNORDERED, null);
  }

  /**
   * Returns an instance which specifies that ordering is guaranteed to be always be the same across
   * iterations, and across releases. Some methods may have stronger guarantees.
   *
   * <p>This instance is only useful in combination with {@code incidentEdgeOrder}, e.g. {@code
   * graphBuilder.incidentEdgeOrder(ElementOrder.stable())}.
   *
