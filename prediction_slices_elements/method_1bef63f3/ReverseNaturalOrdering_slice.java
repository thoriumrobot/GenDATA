// Source-based slice around line 34
// Method: <com.google.common.collect.ReverseNaturalOrdering: int compare(Comparable,Comparable)>

import java.util.Iterator;

/** An ordering that uses the reverse of the natural order of the values. */
@GwtCompatible
final class ReverseNaturalOrdering extends Ordering<Comparable<?>> implements Serializable {
  static final ReverseNaturalOrdering INSTANCE = new ReverseNaturalOrdering();

  @Override
  @SuppressWarnings("unchecked") // TODO(kevinb): the right way to explain this??
  public int compare(Comparable<?> left, Comparable<?> right) {
    checkNotNull(left); // right null is caught later
    if (left == right) {
      return 0;
    }

    return ((Comparable<Object>) right).compareTo(left);
  }

  @Override
  public <S extends Comparable<?>> Ordering<S> reverse() {
