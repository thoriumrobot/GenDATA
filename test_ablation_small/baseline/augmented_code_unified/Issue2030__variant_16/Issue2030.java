    @Positive
public class Issue2030 {
    @Positive
  double roundIntermediate(double x) {
    @Positive
    if (x >= 0.0) {
    @Positive
      return x;
    @Positive
    } else {
    @Positive
      return (long) x - 1;
    @Positive
    }
    @Positive
  }
    @Positive
}
