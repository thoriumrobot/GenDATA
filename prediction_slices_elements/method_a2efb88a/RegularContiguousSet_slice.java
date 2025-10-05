// Source-based slice around line 191
// Method: <com.google.common.collect.RegularContiguousSet: boolean isEmpty()>

    }
  }

  @Override
  public boolean containsAll(Collection<?> targets) {
    return Collections2.containsAllImpl(this, targets);
  }

  @Override
  public boolean isEmpty() {
    return false;
  }

  @Override
  @SuppressWarnings("unchecked") // TODO(cpovirk): Use a shared unsafeCompare method.
  public ContiguousSet<C> intersection(ContiguousSet<C> other) {
    checkNotNull(other);
    checkArgument(this.domain.equals(other.domain));
    if (other.isEmpty()) {
      return other;
