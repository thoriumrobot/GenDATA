import org.checkerframework.checker.index.qual.IndexOrHigh;
    @Positive
import org.checkerframework.checker.index.qual.LTEqLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.Positive;

    @Positive
public class Modulo {
    @Positive
  void m1(Object[] a, @IndexOrHigh("#1") int i, @NonNegative int j) {
    @Positive
    @IndexFor("a") int k = j % i;
    @Positive
  }

    @Positive
  void m1p(Object[] a, @Positive @LTEqLengthOf("#1") int i, @Positive int j) {
    @Positive
    @IndexFor("a") int k = j % i;
