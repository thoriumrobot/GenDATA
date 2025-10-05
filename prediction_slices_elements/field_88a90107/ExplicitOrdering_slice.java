// Source-based slice around line 29
// Method: com.google.common.collect.ExplicitOrdering.rankMap

import com.google.common.annotations.GwtIncompatible;
import com.google.common.annotations.J2ktIncompatible;
import java.io.Serializable;
import java.util.List;
import org.jspecify.annotations.Nullable;

/** An ordering that compares objects according to a given order. */
@GwtCompatible
final class ExplicitOrdering<T> extends Ordering<T> implements Serializable {
  final ImmutableMap<T, Integer> rankMap;

  ExplicitOrdering(List<T> valuesInOrder) {
    this(Maps.indexMap(valuesInOrder));
  }

  ExplicitOrdering(ImmutableMap<T, Integer> rankMap) {
    this.rankMap = rankMap;
  }

  @Override
