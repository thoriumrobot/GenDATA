// Source-based slice around line 220
// Method: <com.google.common.math.StatsAccumulator: Stats snapshot()>

        mean = calculateNewMeanNonFinite(mean, otherMean);
        sumOfSquaresOfDeltas = NaN;
      }
      min = Math.min(min, otherMin);
      max = Math.max(max, otherMax);
    }
  }

  /** Returns an immutable snapshot of the current statistics. */
  public Stats snapshot() {
    return new Stats(count, mean, sumOfSquaresOfDeltas, min, max);
  }

  /** Returns the number of values. */
  public long count() {
    return count;
  }

  /**
   * Returns the <a href="http://en.wikipedia.org/wiki/Arithmetic_mean">arithmetic mean</a> of the
