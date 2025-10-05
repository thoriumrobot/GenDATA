// Source-based slice around line 144
// Method: com.google.common.collect.EvictingQueue.serialVersionUID

     * their users.
     *
     * However, the checker *we* use has this special knowledge about `Collection.toArray()` anyway,
     * so in our implementation code, we can rely on that. That's why the expression below
     * type-checks.
     */
    return super.toArray();
  }

  @GwtIncompatible @J2ktIncompatible private static final long serialVersionUID = 0L;
}
