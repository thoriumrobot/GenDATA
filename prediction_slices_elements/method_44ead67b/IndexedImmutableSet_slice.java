// Source-based slice around line 30
// Method: <com.google.common.collect.IndexedImmutableSet: E get(int)>

import com.google.common.annotations.GwtCompatible;
import com.google.common.annotations.GwtIncompatible;
import com.google.common.annotations.J2ktIncompatible;
import java.util.Spliterator;
import java.util.function.Consumer;
import org.jspecify.annotations.Nullable;

@GwtCompatible
abstract class IndexedImmutableSet<E> extends ImmutableSet.CachingAsList<E> {
  abstract E get(int index);

  @Override
  public UnmodifiableIterator<E> iterator() {
    return asList().iterator();
  }

  @Override
  public Spliterator<E> spliterator() {
    return CollectSpliterators.indexed(size(), SPLITERATOR_CHARACTERISTICS, this::get);
  }
