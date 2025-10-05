// Source-based slice around line 46
// Method: <com.google.common.collect.Cut: BoundType typeAsLowerBound()>

abstract class Cut<C extends Comparable> implements Comparable<Cut<C>>, Serializable {
  final C endpoint;

  Cut(C endpoint) {
    this.endpoint = endpoint;
  }

  abstract boolean isLessThan(C value);

  abstract BoundType typeAsLowerBound();

  abstract BoundType typeAsUpperBound();

  abstract Cut<C> withLowerBoundType(BoundType boundType, DiscreteDomain<C> domain);

  abstract Cut<C> withUpperBoundType(BoundType boundType, DiscreteDomain<C> domain);

  abstract void describeAsLowerBound(StringBuilder sb);

  abstract void describeAsUpperBound(StringBuilder sb);
