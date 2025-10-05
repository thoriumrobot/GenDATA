// Source-based slice around line 29
// Method: com.google.common.base.Absent.INSTANCE

import com.google.common.annotations.GwtIncompatible;
import com.google.common.annotations.J2ktIncompatible;
import java.util.Collections;
import java.util.Set;
import org.jspecify.annotations.Nullable;

/** Implementation of an {@link Optional} not containing a reference. */
@GwtCompatible
final class Absent<T> extends Optional<T> {
  static final Absent<Object> INSTANCE = new Absent<>();

  @SuppressWarnings("unchecked") // implementation is "fully variant"
  static <T> Optional<T> withType() {
    return (Optional<T>) INSTANCE;
  }

  private Absent() {}

  @Override
  public boolean isPresent() {
