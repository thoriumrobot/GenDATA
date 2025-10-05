// Source-based slice around line 48
// Method: com.google.common.math.StatsAccumulator.sumOfSquaresOfDeltas

@GwtIncompatible
public final class StatsAccumulator {
  /** Creates a new accumulator. */
  public StatsAccumulator() {}

  // These fields must satisfy the requirements of Stats' constructor as well as those of the stat
  // methods of this class.
  private long count = 0;
  private double mean = 0.0; // any finite value will do, we only use it to multiply by zero for sum
  private double sumOfSquaresOfDeltas = 0.0;
  private double min = NaN; // any value will do
  private double max = NaN; // any value will do

  /** Adds the given value to the dataset. */
  public void add(double value) {
    if (count == 0) {
      count = 1;
      mean = value;
      min = value;
      max = value;
