    @Positive
import java.util.Date;

    @Positive
public class ArrayLengthLBC {

    @Positive
  public static Date[] add_date(Date[] dates, Date new_date) {
    @Positive
    Date[] new_dates = new Date[dates.length + 1];
    @Positive
    System.arraycopy(dates, 0, new_dates, 0, dates.length);
    @Positive
    new_dates[dates.length] = new_date;
    @Positive
    Date[] new_dates_cast = new_dates;
    @Positive
    return (new_dates_cast);
    @Positive
  }
    @Positive
}
// a comment

// CFWR semantic augmentation - variant 0
