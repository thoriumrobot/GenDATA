// Source-based slice around line 36
// Method: <com.google.common.base.Present: boolean isPresent()>

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

  @Override
  public T get() {
    return reference;
  }

  @Override
  public T or(T defaultValue) {
