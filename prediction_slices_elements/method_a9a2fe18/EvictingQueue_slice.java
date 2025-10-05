// Source-based slice around line 130
// Method: <com.google.common.collect.EvictingQueue: Object[] toArray()>

    if (size >= maxSize) {
      clear();
      return Iterables.addAll(this, Iterables.skip(collection, size - maxSize));
    }
    return standardAddAll(collection);
  }

  @Override
  @J2ktIncompatible // Incompatible return type change. Use inherited implementation
  public Object[] toArray() {
    /*
     * If we could, we'd declare the no-arg `Collection.toArray()` to return "Object[] but elements
     * have the same nullness as E." Since we can't, we declare it to return nullable elements, and
     * we can override it in our non-null-guaranteeing subtypes to present a better signature to
     * their users.
     *
     * However, the checker *we* use has this special knowledge about `Collection.toArray()` anyway,
     * so in our implementation code, we can rely on that. That's why the expression below
     * type-checks.
     */
