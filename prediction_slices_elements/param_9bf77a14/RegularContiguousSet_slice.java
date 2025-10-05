// Source-based slice around line 197
// Method: <com.google.common.collect.RegularContiguousSet: ContiguousSet intersection(ContiguousSet)>

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
    } else {
      C lowerEndpoint = Ordering.<C>natural().max(this.first(), other.first());
      C upperEndpoint = Ordering.<C>natural().min(this.last(), other.last());
      return (lowerEndpoint.compareTo(upperEndpoint) <= 0)
          ? ContiguousSet.create(Range.closed(lowerEndpoint, upperEndpoint), domain)
          : new EmptyContiguousSet<C>(domain);
