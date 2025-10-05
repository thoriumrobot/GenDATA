/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.IndexFor;
    @Positive
import org.checkerframework.checker.index.qual.LTEqLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;

// @skip-test until we bring list support back

    @Positive
public class ListAddAll {

    @Positive
  List<Integer> listField;
    @Positive
  List<Integer> coll;

    @Positive
  void ListAddAll(
    @Positive
      @LTLengthOf("#3") int index, @LTEqLengthOf("#3") int notIndex, List<Integer> list) {
    @Positive
    list.addAll(index, coll);

    // :: error: (list.access.unsafe.high)
    @Positive
    list.addAll(notIndex, coll);
    @Positive
  }

    @Positive
  int[] arr = {0};

    @Positive
  void ListAddAllWrongName(@LTLengthOf("arr") int index, List<Integer> list) {
    // :: error: (list.access.unsafe.high)
    @Positive
    list.addAll(index, coll);
    @Positive
  }

    @Positive
  void ListAddAllField() {
    @Positive
    listField.addAll(listField.size() - 1, coll);
    @Positive
    listField.addAll(this.listField.size() - 1, coll);
    @Positive
    this.listField.addAll(listField.size() - 1, coll);
    @Positive
    this.listField.addAll(this.listField.size() - 1, coll);

    // :: error: (list.access.unsafe.high)
    @Positive
    listField.addAll(listField.size(), coll);
    // :: error: (list.access.unsafe.high)
    @Positive
    listField.addAll(this.listField.size(), coll);
    // :: error: (list.access.unsafe.high)
    @Positive
    this.listField.addAll(listField.size(), coll);
    // :: error: (list.access.unsafe.high)
    @Positive
    this.listField.addAll(this.listField.size(), coll);
    @Positive
  }

    @Positive
  void ListAddAllFieldUserAnnotation(@IndexFor("listField") int i) {
    @Positive
    listField.addAll(i, coll);
    @Positive
    this.listField.addAll(i, coll);

    // :: error: (list.access.unsafe.high)
    @Positive
    listField.addAll(1 + i, coll);
    // :: error: (list.access.unsafe.high)
    @Positive
    this.listField.addAll(1 + i, coll);
    @Positive
  }

    @Positive
  void ListAddAllUserAnnotation(@IndexFor("#2") int i, List<Integer> list) {
    @Positive
    list.addAll(i, coll);

    // :: error: (list.access.unsafe.high)
    @Positive
    list.addAll(1 + i, coll);
    @Positive
  }
    @Positive
}
