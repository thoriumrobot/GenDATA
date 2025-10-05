/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import java.text.DecimalFormat;
    @Positive
import org.checkerframework.checker.index.qual.IndexFor;

    @Positive
public final class Stopwatch {
    @Positive
  private static final DecimalFormat[] timeFormat = {
    @Positive
    new DecimalFormat("#.#"),
    @Positive
    new DecimalFormat("#.#"),
    @Positive
    new DecimalFormat("#.#"),
    @Positive
    new DecimalFormat("#.#"),
    @Positive
    new DecimalFormat("#.#"),
    @Positive
  };

    @Positive
  public DecimalFormat format(@IndexFor("Stopwatch.timeFormat") int digits) {
    @Positive
    return Stopwatch.timeFormat[digits];
    @Positive
  }
    @Positive
}
