// Source-based slice around line 29
// Method: com.google.common.base.Present.reference

import com.google.common.annotations.GwtIncompatible;
import com.google.common.annotations.J2ktIncompatible;
import java.util.Collections;
import java.util.Set;
import org.jspecify.annotations.Nullable;

/** Implementation of an {@link Optional} containing a reference. */
@GwtCompatible
final class Present<T> extends Optional<T> {
  private final T reference;

  Present(T reference) {
    this.reference = reference;
  }

  @Override
  public boolean isPresent() {
    return true;
  }

